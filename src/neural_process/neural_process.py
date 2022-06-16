import copy
import logging
import os
import types
from itertools import product
from typing import List, Optional, Tuple

import numpy as np
import torch
import yaml
from metalearning_benchmarks.base_benchmark import (
    MetaLearningBenchmark,
    MetaLearningTask,
)
from np_util.datasets import MetaLearningDataset
from np_util.tqdm_logging_handler import TqdmLoggingHandler
from torch.utils.data import DataLoader
from tqdm import tqdm

from neural_process.aggregator import (
    BayesianAggregator,
    MaxAggregator,
    MaxAggregatorRtoZ,
    MeanAggregator,
    MeanAggregatorRtoZ,
)
from neural_process.decoder_network import DecoderNetworkPB, DecoderNetworkSamples
from neural_process.encoder_network import (
    EncoderNetworkBA,
    EncoderNetworkMA,
    EncoderNetworkSAMA,
)


class NeuralProcess:
    _f_normalizers = "000_normalizers.txt"
    _f_settings = "000_settings.txt"
    _f_n_tasks_seen = "000_n_tasks_seen.txt"
    _available_aggregator_types = ["BA", "MA", "SA_MA", "MAX"]
    _available_loss_types = ["PB", "VI", "MC", "IWMC"]
    _available_input_mlp_std_y = ["xz", "x", "z", "cov_z", ""]
    _available_self_attention_types = [
        None,
        "uniform",
        "laplace",
        "dot_product",
        "multihead",
    ]

    def __init__(
        self,
        logpath: str,
        seed: int,
        d_x: int,
        d_y: int,
        d_z: int,
        n_context: Tuple,
        aggregator_type: str = "BA",
        loss_type: str = "MC",
        input_mlp_std_y: Optional[str] = None,
        self_attention_type: Optional[str] = None,
        latent_prior_scale: float = 1.0,
        f_act: str = "relu",
        n_hidden_layers: int = 2,
        n_hidden_units: int = 16,
        decoder_output_scale: float = 1.0,
        device: str = "cpu",
        adam_lr: float = 1e-4,
        batch_size: int = 16,
        n_samples: int = 16,
    ):
        # build config
        self._config = self._build_config(
            logpath=logpath,
            seed=seed,
            d_x=d_x,
            d_y=d_y,
            d_z=d_z,
            n_context=n_context,
            aggregator_type=aggregator_type,
            loss_type=loss_type,
            input_mlp_std_y=input_mlp_std_y,
            self_attention_type=self_attention_type,
            latent_prior_scale=latent_prior_scale,
            f_act=f_act,
            n_hidden_layers=n_hidden_layers,
            n_hidden_units=n_hidden_units,
            decoder_output_scale=decoder_output_scale,
            device=device,
            adam_lr=adam_lr,
            batch_size=batch_size,
            n_samples=n_samples,
        )

        # logging
        assert os.path.isdir(logpath)
        self._logpath = logpath
        self._logger = None
        self._configure_logger()

        # write config to file
        self._write_config_to_file()

        # initialize random number generator
        self._rng = np.random.RandomState()
        self._seed(self._config["seed"])

        # set n_meta_tasks_seen and write it to file
        self._n_meta_tasks_seen = 0
        self._write_n_meta_tasks_seen_to_file()

        # initialize architecture
        self._modules = []
        self._create_architecture()
        self._set_device("cpu")

        # initialize optimizer
        self._optimizer = None
        self._create_optimizer()

        self._normalizers = {"x_mu": None, "x_std": None, "y_mu": None, "y_std": None}
        self._logger.info("Initialized new model of type {}...".format(type(self)))

    @staticmethod
    def get_valid_model_specs() -> List[dict]:
        model_specs = []
        for (
            aggregator_type,
            loss_type,
            input_mlp_std_y,
            self_attention_type,
        ) in product(
            NeuralProcess._available_aggregator_types,
            NeuralProcess._available_loss_types,
            NeuralProcess._available_input_mlp_std_y,
            NeuralProcess._available_self_attention_types,
        ):
            if (loss_type == "PB") and (input_mlp_std_y != "xz"):
                # Deterministic architectures require to pass "xz" to variance decoder
                # TODO: actually, this should be called mu_z, cov_z for BA
                continue
            if (aggregator_type != "SA_MA") and (self_attention_type is not None):
                continue
            if (aggregator_type == "SA_MA") and (self_attention_type is None):
                continue
            model_specs.append(
                {
                    "aggregator_type": aggregator_type,
                    "loss_type": loss_type,
                    "input_mlp_std_y": input_mlp_std_y,
                    "self_attention_type": self_attention_type,
                }
            )
        return model_specs

    @staticmethod
    def is_valid_model_spec(model_spec: dict) -> bool:
        ms = copy.deepcopy(model_spec)
        if "self_attention_type" not in ms:
            ms["self_attention_type"] = None
        return ms in NeuralProcess.get_valid_model_specs()

    @staticmethod
    def _build_config(
        logpath: int,
        seed: int,
        d_x: int,
        d_y: int,
        d_z: int,
        n_context: Tuple,
        aggregator_type: str,
        loss_type: str,
        input_mlp_std_y: Optional[str],
        self_attention_type: Optional[str],
        f_act: str,
        n_hidden_layers: int,
        n_hidden_units: int,
        latent_prior_scale: float,
        decoder_output_scale: float,
        device: str,
        adam_lr: float,
        batch_size: int,
        n_samples: int,
    ) -> dict:
        config = {
            "logpath": logpath,
            "seed": seed,
            "d_x": d_x,
            "d_y": d_y,
            "d_z": d_z,
            "aggregator_type": aggregator_type,
            "loss_type": loss_type,
            "input_mlp_std_y": input_mlp_std_y,
            "self_attention_type": self_attention_type,
            "f_act": f_act,
            "n_hidden_layers": n_hidden_layers,
            "n_hidden_units": n_hidden_units,
            "adam_lr": adam_lr,
            "batch_size": batch_size,
            "n_context_meta": n_context,
            "n_context_val": n_context,
            "device": device,
        }

        # check model spec
        model_spec = {
            "aggregator_type": config["aggregator_type"],
            "loss_type": config["loss_type"],
            "input_mlp_std_y": config["input_mlp_std_y"],
            "self_attention_type": config["self_attention_type"],
        }
        assert NeuralProcess.is_valid_model_spec(model_spec)

        # network architecture for encoder and decoder networks
        network_arch = config["n_hidden_layers"] * [config["n_hidden_units"]]

        # decoder kwargs
        decoder_kwargs = {
            "mlp_layers_mu_y": network_arch,
            "input_mlp_std_y": config["input_mlp_std_y"],
        }
        if config["aggregator_type"] == "BA" and config["loss_type"] == "PB":
            decoder_kwargs["arch"] = "separate_networks_separate_input"
        else:  # VI or MC or IWMC
            decoder_kwargs["arch"] = "separate_networks"
        if config["input_mlp_std_y"] != "":
            decoder_kwargs["mlp_layers_std_y"] = network_arch
        else:
            decoder_kwargs["global_std_y"] = decoder_output_scale
            decoder_kwargs["global_std_y_is_learnable"] = False

        # encoder_kwargs
        if config["aggregator_type"] == "BA":
            encoder_kwargs = {
                "arch": "separate_networks",
                "mlp_layers_r": network_arch,
                "mlp_layers_cov_r": network_arch,
            }
        elif config["aggregator_type"] == "SA_MA":
            assert config["self_attention_type"] is not None
            encoder_kwargs = {
                "mlp_layers_r": network_arch,
                "self_attention_type": config["self_attention_type"],
            }
            if config["self_attention_type"] == "multihead":
                num_heads = 8
                assert (
                    config["d_z"] % num_heads == 0
                ), "d_z has to be divisible by num_heads = {:d}".format(num_heads)
                encoder_kwargs["num_heads"] = num_heads
        else:  # MA or MAX
            encoder_kwargs = {"mlp_layers_r": network_arch}

        # aggregator_kwargs
        if (
            config["aggregator_type"] == "MA"
            or config["aggregator_type"] == "MAX"
            or config["aggregator_type"] == "SA_MA"
        ) and (config["loss_type"] != "PB"):
            aggregator_kwargs = {
                "arch": "separate_networks",
                "mlp_layers_r_to_mu_z": 1 * [config["n_hidden_units"]],
                "mlp_layers_r_to_cov_z": 1 * [config["n_hidden_units"]],
            }
        elif config["aggregator_type"] == "BA":
            aggregator_kwargs = {
                "var_z_0": latent_prior_scale,
                "var_z_0_is_learnable": False,
            }
        else:
            aggregator_kwargs = {}

        # loss_kwargs
        if config["loss_type"] == "MC" or config["loss_type"] == "IWMC":
            loss_kwargs = {"n_marg": n_samples}
        else:  # loss_type == "PB" or loss_type == "VI"
            loss_kwargs = {}

        config.update(
            {
                "encoder_kwargs": encoder_kwargs,
                "aggregator_kwargs": aggregator_kwargs,
                "decoder_kwargs": decoder_kwargs,
                "loss_kwargs": loss_kwargs,
                "predictions_are_deterministic": config["loss_type"] == "PB",
            }
        )

        return config

    @property
    def settings(self) -> dict:
        return self._config

    @property
    def n_meta_tasks_seen(self) -> int:
        return self._n_meta_tasks_seen

    @property
    def parameters(self):
        """
        Returns an iterable of parameters that are trainable in the model.
        """
        parameters = []
        for module in self._modules:
            if isinstance(module.parameters, list):
                parameters += module.parameters
            else:
                assert isinstance(module.parameters(), types.GeneratorType)
                parameters += list(module.parameters())
        return parameters

    @property
    def _normalizers_available(self):
        res = True

        if self._normalizers["x_mu"] is None:
            res = False
        if self._normalizers["x_std"] is None:
            res = False
        if self._normalizers["y_mu"] is None:
            res = False
        if self._normalizers["y_std"] is None:
            res = False

        return res

    def _configure_logger(self):
        """
        Creates a logger and pairs it with the tqdm-progress logging handler.
        """
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)

        # define format
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # write to stderr
        # the logger might already have a TqdmLoggingHandler
        has_sh = any(
            [
                isinstance(handler, TqdmLoggingHandler)
                for handler in self._logger.handlers
            ]
        )
        if not has_sh:
            sh = TqdmLoggingHandler()
            sh.setLevel(logging.DEBUG)
            sh.setFormatter(formatter)
            self._logger.addHandler(sh)

    def _create_architecture(self) -> None:
        # create encoder
        if self._config["aggregator_type"] == "BA":
            encoder = EncoderNetworkBA
        elif self._config["aggregator_type"] == "SA_MA":
            encoder = EncoderNetworkSAMA
        else:  # MeanAggregator or MaxAggregator
            encoder = EncoderNetworkMA
        self.encoder = encoder(
            d_x=self._config["d_x"],
            d_y=self._config["d_y"],
            d_r=self._config["d_z"],  # we set d_r == d_z
            f_act=self._config["f_act"],
            seed=self._config["seed"],
            **self._config["encoder_kwargs"],
        )

        # create aggregator
        if self._config["aggregator_type"] == "BA":
            aggregator = BayesianAggregator
        elif self._config["aggregator_type"] == "MAX":
            if self._config["loss_type"] == "PB":
                aggregator = MaxAggregator
            else:  # MonteCarlo or VI-inspired losses
                aggregator = MaxAggregatorRtoZ
        else:  # MeanAggregator (w/ or w/o self-attention)
            if self._config["loss_type"] == "PB":
                aggregator = MeanAggregator
            else:  # MonteCarlo or VI-inspired losses
                aggregator = MeanAggregatorRtoZ
        self.aggregator = aggregator(
            d_r=self._config["d_z"],  # we set d_r == d_z
            d_z=self._config["d_z"],
            f_act=self._config["f_act"],
            seed=self._config["seed"],
            **self._config["aggregator_kwargs"],
        )

        # create decoder
        if self._config["aggregator_type"] == "BA":
            if self._config["loss_type"] == "PB":
                decoder = DecoderNetworkPB
            else:  # MonteCarlo or VI-inspired loss
                decoder = DecoderNetworkSamples
        else:  # MeanAggregator (w/ or w/o self-attention)
            # MA + PB also uses DecoderNetworkSamples, as the z-input can be used for r
            decoder = DecoderNetworkSamples
        self.decoder = decoder(
            d_x=self._config["d_x"],
            d_y=self._config["d_y"],
            d_z=self._config["d_z"],
            f_act=self._config["f_act"],
            seed=self._config["seed"],
            **self._config["decoder_kwargs"],
        )

        self._modules = [self.encoder, self.aggregator, self.decoder]

    def _create_optimizer(self) -> None:
        self._optimizer = torch.optim.Adam(
            params=self.parameters, lr=self._config["adam_lr"]
        )

    def _set_device(self, device):
        if device == "cuda" and not torch.cuda.is_available():
            self._logger.warning("CUDA not available! Using CPU instead!")
            self.device = "cpu"
        else:
            self.device = device

        for module in self._modules:
            module.to(self.device)

    def _write_config_to_file(self):
        with open(os.path.join(self._logpath, self._f_settings), "w") as f:
            yaml.safe_dump(self._config, f)

    def _load_config_from_file(self):
        with open(os.path.join(self._logpath, self._f_settings), "r") as f:
            config = yaml.safe_load(f)

        return config

    def _write_n_meta_tasks_seen_to_file(self):
        with open(os.path.join(self._logpath, self._f_n_tasks_seen), "w") as f:
            yaml.safe_dump(self._n_meta_tasks_seen, f)

    def _load_n_meta_tasks_seen_from_file(self):
        with open(os.path.join(self._logpath, self._f_n_tasks_seen), "r") as f:
            epoch = yaml.safe_load(f)

        return epoch

    def _load_weights_from_file(self):
        for module in self._modules:
            module.load_weights(self._logpath, self._n_meta_tasks_seen)

    def _write_normalizers_to_file(self):
        # do this only once right at the beginning
        assert self._n_meta_tasks_seen == 0

        normalizers_as_lists = self._normalizers.copy()
        for (key, val) in normalizers_as_lists.items():
            normalizers_as_lists[key] = val.to("cpu").tolist()

        with open(os.path.join(self._logpath, self._f_normalizers), "w") as f:
            yaml.safe_dump(normalizers_as_lists, f)

    def _load_normalizers_from_file(self):
        with open(os.path.join(self._logpath, self._f_normalizers), "r") as f:
            self._normalizers = yaml.safe_load(f)
        for (key, val) in self._normalizers.items():
            self._normalizers[key] = torch.tensor(val)

    def _determine_normalizers(self, benchmark, n_tasks=1000):
        # check that we've not already determined normalizers
        assert not self._normalizers_available
        self._logger.info(
            "Computing normalizers on data from {:d} tasks...".format(n_tasks)
        )

        # create dataloader
        dataloader = DataLoader(
            dataset=MetaLearningDataset(benchmark),
            batch_size=n_tasks,
            collate_fn=lambda task_list: self._collate_batch(task_list),
        )

        # load n_tasks tasks
        x, y = next(iter(dataloader))

        # compute normalizers across (n_task, n_datapoints_per_task) dimensions
        self._normalizers["x_mu"] = x.double().mean(dim=(0, 1)).float()
        self._normalizers["y_mu"] = y.double().mean(dim=(0, 1)).float()
        self._normalizers["x_std"] = x.double().std(dim=(0, 1)).float()
        self._normalizers["y_std"] = y.double().std(dim=(0, 1)).float()

        self._write_normalizers_to_file()

    def _normalize_x(self, x):
        assert x.shape[-1] == self._normalizers["x_mu"].shape[0]
        if not self._normalizers_available:
            raise RuntimeError(
                "Normalizers not available! Maybe model has not been trained yet?"
            )

        n_dim_to_expand = x.ndim - 1
        # broadcast normalizers to data shape
        expander = (None,) * n_dim_to_expand + (Ellipsis,)
        normalizer_mu = self._normalizers["x_mu"][expander].to(self.device)
        normalizer_std = self._normalizers["x_std"][expander].to(self.device)

        x_normalized = x - normalizer_mu
        if (normalizer_std != 0.0).all():
            x_normalized /= normalizer_std

        return x_normalized

    def _normalize_y(self, y):
        assert y.shape[-1] == self._normalizers["y_mu"].shape[0]
        if not self._normalizers_available:
            raise RuntimeError(
                "Normalizers not available! Maybe model has not been trained yet?"
            )

        n_dim_to_expand = y.ndim - 1
        expander = (None,) * n_dim_to_expand + (Ellipsis,)
        normalizer_mu = self._normalizers["y_mu"][expander].to(self.device)
        normalizer_std = self._normalizers["y_std"][expander].to(self.device)

        y_normalized = y - normalizer_mu
        if (normalizer_std != 0.0).all():
            y_normalized /= normalizer_std

        return y_normalized

    def _denormalize_mu_y(self, mu_y):
        assert mu_y.shape[-1] == self._normalizers["y_mu"].shape[0]

        n_dim_to_expand = mu_y.ndim - 1
        expander = (None,) * n_dim_to_expand + (Ellipsis,)
        normalizer_mu = self._normalizers["y_mu"][expander]
        normalizer_std = self._normalizers["y_std"][expander]

        normalizer_mu = normalizer_mu.to(self.device)
        normalizer_std = normalizer_std.to(self.device)

        mu_y_denormalized = mu_y
        if (normalizer_std != 0.0).all():
            mu_y_denormalized *= normalizer_std
        mu_y_denormalized += normalizer_mu

        return mu_y_denormalized.reshape(mu_y.shape)

    def _denormalize_std_y(self, std_y):
        assert std_y.shape[-1] == self._normalizers["y_std"].shape[0]

        n_dim_to_expand = std_y.ndim - 1
        expander = (None,) * n_dim_to_expand + (Ellipsis,)
        normalizer_std = self._normalizers["y_std"][expander]

        std_y_denormalized = std_y
        if (normalizer_std != 0.0).all():
            std_y_denormalized *= normalizer_std

        return std_y_denormalized.reshape(std_y.shape)

    def _seed(self, seed: int) -> None:
        self._rng.seed(seed=seed)

    def _collate_batch(
        self,
        task_list: List[MetaLearningTask],
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # collect sizes
        n_tsk = len(task_list)
        n_points_per_task = task_list[0].n_points
        d_x = task_list[0].d_x
        d_y = task_list[0].d_y

        # collect all tasks
        x = torch.zeros((n_tsk, n_points_per_task, d_x))
        y = torch.zeros((n_tsk, n_points_per_task, d_y))
        for i, task in enumerate(task_list):
            x[i] = torch.tensor(task.x)
            y[i] = torch.tensor(task.y)

        # send to device
        x, y = x.to(self.device), y.to(self.device)

        return x, y

    def _compute_loss(
        self, x: torch.Tensor, y: torch.Tensor, mode: str
    ) -> torch.Tensor:
        # determine loss_type, loss_kwargs and ctx_steps
        if mode == "meta":
            # use the loss_type and loss_kwargs given in settings
            loss_type = self._config["loss_type"]
            loss_kwargs = self._config["loss_kwargs"]

            # sample a random context set size
            ctx_steps = [
                self._rng.randint(
                    low=self._config["n_context_meta"][0],
                    high=self._config["n_context_meta"][1] + 1,
                    size=(1,),
                ).item()
            ]
            n_ctx = max(ctx_steps)
        elif mode == "val":
            # TODO: is it correct to evaluate MC also for IWMC-models?
            loss_type = "MC"
            loss_kwargs = {
                "n_marg": 1
                if self._config["predictions_are_deterministic"]
                else 500,  # TODO: how many samples to use?
            }

            # average over some context set sizes
            ctx_steps = list(
                np.linspace(
                    self.settings["n_context_val"][0],
                    self.settings["n_context_val"][1],
                    num=min(
                        5,
                        (
                            self.settings["n_context_val"][1]
                            - self.settings["n_context_val"][0]
                        )
                        + 1,
                    ),  # TODO: how many context steps to use?
                    dtype=np.int,
                )
            )
            n_ctx = max(ctx_steps)
        else:
            raise ValueError("Unknown value of argument 'mode'!")

        # create context and test sets
        x_ctx, y_ctx, x_tgt, y_tgt, latent_obs_all = self._create_ctx_tst_sets(
            x_all=x, y_all=y, n_ctx=n_ctx, loss_type=loss_type
        )

        # compute loss for all context set sizes
        loss = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        for i in range(len(ctx_steps)):
            # reset aggregator
            self.aggregator.reset(x_ctx.shape[0])

            # encode current context set
            # (due to self-attention, we cannot encode the whole context set at once)
            cur_latent_obs = self.encoder.encode(
                x=x_ctx[:, : ctx_steps[i], :], y=y_ctx[:, : ctx_steps[i], :]
            )
            # due to self-attention, we cannot use incremental updates
            self.aggregator.update(latent_obs=cur_latent_obs, incremental=False)

            # compute loss
            ls_ctx = self.aggregator.last_latent_state
            agg_state_ctx = self.aggregator.last_agg_state

            # TODO: remove ls-dimension altogether for backwards compatibility
            # add dummy ls-dimension
            ls_ctx_dummy = []
            for i in range(len(ls_ctx)):
                if ls_ctx[i] is not None:
                    ls_ctx_dummy.append(ls_ctx[i][:, None, :])
                else:
                    ls_ctx_dummy.append(None)
            ls_ctx = tuple(ls_ctx_dummy)
            agg_state_ctx = tuple(
                agg_state_ctx[i][:, None, :] for i in range(len(agg_state_ctx))
            )

            # decode latent state
            mu_z_ctx = ls_ctx[0]
            cov_z_ctx = ls_ctx[1]
            assert mu_z_ctx.ndim == 3
            if cov_z_ctx is not None:
                assert cov_z_ctx.ndim == 3
                assert mu_z_ctx.shape[1] == cov_z_ctx.shape[1] == 1

            if loss_type == "PB":
                loss = loss - self._conditional_ll(
                    x_tgt=x_tgt,
                    y_tgt=y_tgt,
                    mu_z=mu_z_ctx,
                    cov_z=cov_z_ctx,
                )
            elif loss_type == "MC":
                loss = loss - self._log_marg_lhd_np_mc(
                    x_tgt=x_tgt,
                    y_tgt=y_tgt,
                    mu_z_ctx=mu_z_ctx,
                    cov_z_ctx=cov_z_ctx,
                    n_marg=loss_kwargs["n_marg"],
                )
            elif loss_type == "IWMC":
                initial_latent_state = self.aggregator.initial_latent_state
                mu_z_prior = initial_latent_state[0]
                cov_z_prior = initial_latent_state[1]
                loss = loss - self._log_marg_lhd_true_np_mc(
                    x_tgt=x_tgt,
                    y_tgt=y_tgt,
                    x_ctx=x_ctx,
                    y_ctx=y_ctx,
                    mu_z_ctx=mu_z_ctx,
                    cov_z_ctx=cov_z_ctx,
                    mu_z_prior=mu_z_prior,
                    cov_z_prior=cov_z_prior,
                    n_marg=loss_kwargs["n_marg"],
                )
            elif loss_type == "VI":
                loss = loss - self._elbo_np(
                    x_tgt=x_tgt,
                    y_tgt=y_tgt,
                    mu_z_ctx=mu_z_ctx,
                    cov_z_ctx=cov_z_ctx,
                    agg_state_ctx=agg_state_ctx,
                    latent_obs_all=latent_obs_all,
                )
            else:
                raise ValueError("Unknown loss_type '{}'!".format(loss_type))

        # average loss over number of context sets evaluated
        loss = loss / len(ctx_steps)

        return loss

    def save_model(self):
        self._logger.info("Saving model...")
        for module in self._modules:
            module.save_weights(self._logpath, self._n_meta_tasks_seen)
        self._write_n_meta_tasks_seen_to_file()

    def load_model(self, load_n_meta_tasks_seen: int = -1) -> None:
        assert isinstance(load_n_meta_tasks_seen, int)

        # load settings
        config = self._load_config_from_file()
        assert config == self._config

        # load n_meta_tasks_seen
        if load_n_meta_tasks_seen == -1:  # load latest checkpoint
            self._n_meta_tasks_seen = self._load_n_meta_tasks_seen_from_file()
        else:
            self._n_meta_tasks_seen = load_n_meta_tasks_seen

        self._logger.info(
            "Loaded model at n_tasks_seen={:d}!".format(self._n_meta_tasks_seen)
        )

        # load architecture
        self._create_architecture()
        self._load_weights_from_file()
        self._load_normalizers_from_file()
        self._set_device("cpu")

        # initialize random number generator
        self._rng = np.random.RandomState()
        self._seed(self._config["seed"])

        # initialize optimizer
        self._create_optimizer()

        self._is_initialized = True

    def _check_data_shapes(self, x, y=None):
        if len(x.shape) < 2 or x.shape[-1] != self.settings["d_x"]:
            raise NotImplementedError("x has wrong shape!")

        if y is not None:
            if len(y.shape) < 2 or y.shape[-1] != self.settings["d_y"]:
                raise NotImplementedError("y has wrong shape!")

    def _prepare_data_for_testing(self, data):
        data = torch.Tensor(data)
        assert 2 <= data.ndim <= 3
        if data.ndim == 2:
            data = data[None, :, :]  # add task dimension
        data.to(self.device)

        return data

    def _create_ctx_tst_sets(self, x_all, y_all, n_ctx, loss_type):
        n_all = x_all.shape[1]

        # determine context points
        idx_pts = self._rng.permutation(x_all.shape[1])
        x_ctx = x_all[:, idx_pts[:n_ctx], :]
        y_ctx = y_all[:, idx_pts[:n_ctx], :]

        if not (loss_type == "MCIW" or loss_type == "IWMCIW" or loss_type == "VI"):
            # use remaining points as test points
            x_tgt = x_all[:, idx_pts[n_ctx:], :]
            y_tgt = y_all[:, idx_pts[n_ctx:], :]
            latent_obs_all = None  # not necessary
        else:  # loss_type == "VI"
            # sample a test set from the remaining points
            n_tst = self._rng.randint(low=1, high=n_all - n_ctx + 1, size=(1,)).squeeze()
            x_tgt = x_all[:, idx_pts[n_ctx : n_ctx + n_tst], :]
            y_tgt = y_all[:, idx_pts[n_ctx : n_ctx + n_tst], :]
            latent_obs_all = self.encoder.encode(x_all, y_all)

        return x_ctx, y_ctx, x_tgt, y_tgt, latent_obs_all

    def _conditional_ll(self, x_tgt, y_tgt, mu_z, cov_z):
        assert x_tgt.ndim == y_tgt.ndim == 3  # (n_tsk, n_tst, d_x/d_y)
        assert x_tgt.nelement() != 0
        assert y_tgt.nelement() != 0

        # obtain predictions
        mu_y, std_y = self._predict(x=x_tgt, mu_z=mu_z, cov_z=cov_z, n_marg=1)
        # mu_y, std_y shape = (n_tsk, n_ls, 1, n_tst, d_y)
        mu_y, std_y = mu_y.squeeze(2), std_y.squeeze(2)
        assert mu_y.ndim == 4 and std_y.ndim == 4

        # add latent state dimension to y-values
        n_ls = mu_y.shape[1]
        y_tgt = y_tgt[:, None, :, :].expand(-1, n_ls, -1, -1)

        # compute mean log-likelihood
        gaussian = torch.distributions.Normal(mu_y, std_y)
        ll = gaussian.log_prob(y_tgt)

        # take sum of lls over output dimension
        ll = torch.sum(ll, axis=-1)

        # take mean over all datapoints
        ll = torch.mean(ll)

        return ll

    def _log_marg_lhd_np_mc(self, x_tgt, y_tgt, mu_z_ctx, cov_z_ctx, n_marg):
        assert x_tgt.ndim == y_tgt.ndim == 3  # (n_tsk, n_tst, d_x/d_y)
        assert x_tgt.nelement() != 0
        assert y_tgt.nelement() != 0

        # obtain predictions
        mu_y, std_y = self._predict(x_tgt, mu_z_ctx, cov_z_ctx, n_marg=n_marg)
        # mu_y, std_y shape = (n_tsk, n_ls, n_marg, n_tst, d_y)

        # add latent state and marginalization dimension to y-values
        n_ls = mu_y.shape[1]
        n_tsk = x_tgt.shape[0]
        n_tgt = x_tgt.shape[1]
        assert n_marg > 0
        y_tgt = y_tgt[:, None, None, :, :].expand(-1, n_ls, n_marg, -1, -1)

        # compute log-likelihood for all datapoints
        gaussian = torch.distributions.Normal(mu_y, std_y)
        ll = gaussian.log_prob(y_tgt)

        # sum log-likelihood over output and datapoint dimension
        ll = torch.sum(ll, dim=(-2, -1))

        # compute MC-average
        ll = torch.logsumexp(ll, dim=2)

        # add -log(n_marg)
        ll = -np.log(n_marg) + ll

        # sum task- and ls-dimensions
        ll = torch.sum(ll, dim=(0, 1))
        assert ll.ndim == 0

        # compute average log-likelihood over all datapoints
        ll = ll / (n_tsk * n_ls * n_tgt)

        return ll

    def _log_marg_lhd_true_np_mc(
        self,
        x_tgt,
        y_tgt,
        x_ctx,
        y_ctx,
        mu_z_ctx,
        cov_z_ctx,
        mu_z_prior,
        cov_z_prior,
        n_marg,
    ):
        assert x_tgt.ndim == y_tgt.ndim == 3  # (n_tsk, n_tst, d_x/d_y)
        assert x_ctx.ndim == y_ctx.ndim == 3  # (n_tsk, n_ctx, d_x/d_y)
        assert x_tgt.shape[0] == x_ctx.shape[0] == y_tgt.shape[0] == y_ctx.shape[0]
        assert x_tgt.shape[2] == x_ctx.shape[2]
        assert x_tgt.shape[2] == x_ctx.shape[2]
        assert y_tgt.shape[2] == y_ctx.shape[2]
        assert x_tgt.nelement() != 0
        assert y_tgt.nelement() != 0
        assert n_marg > 0
        assert mu_z_prior.ndim == cov_z_prior.ndim == 2  # (n_tsk, d_z)
        assert mu_z_ctx.ndim == cov_z_ctx.ndim == 3  # (n_tsk, n_ls, d_z)
        assert mu_z_prior.shape[0] == mu_z_ctx.shape[0] == x_tgt.shape[0]
        assert cov_z_prior.shape[0] == cov_z_ctx.shape[0] == x_tgt.shape[0]

        # shapes
        n_tsk = x_tgt.shape[0]
        n_tgt = x_tgt.shape[1]
        n_ls = mu_z_ctx.shape[1]

        # obtain predictions (use same sample set for both tgt and ctx)
        all_x = torch.cat((x_tgt, x_ctx), dim=1)
        all_mu_y, all_std_y, z = self._predict(
            x=all_x,
            mu_z=mu_z_ctx,
            cov_z=cov_z_ctx,
            n_marg=n_marg,
            return_latent_samples=True,
        )
        # all_mu_y, all_std_y shape = (n_tsk, n_ls, n_marg, n_tst, d_y)
        # z = (n_tsk, n_ls, n_marg, d_z)
        mu_y_tgt, mu_y_ctx = all_mu_y[:, :, :, :n_tgt, :], all_mu_y[:, :, :, n_tgt:, :]
        std_y_tgt, std_y_ctx = (
            all_std_y[:, :, :, :n_tgt, :],
            all_std_y[:, :, :, n_tgt:, :],
        )

        # add latent state and marginalization dimension to y-values
        y_tgt = y_tgt[:, None, None, :, :].expand(-1, n_ls, n_marg, -1, -1)
        y_ctx = y_ctx[:, None, None, :, :].expand(-1, n_ls, n_marg, -1, -1)

        # compute log-likelihood for all datapoints
        gaussian_y_tgt = torch.distributions.Normal(mu_y_tgt, std_y_tgt)
        log_p_y_tgt = gaussian_y_tgt.log_prob(
            y_tgt
        )  # (n_tsk, n_ls, n_marg, n_tgt, d_y)
        gaussian_y_ctx = torch.distributions.Normal(mu_y_ctx, std_y_ctx)
        log_p_y_ctx = gaussian_y_ctx.log_prob(
            y_ctx
        )  # (n_tsk, n_ls, n_marg, n_ctx, d_y)

        # sum log-likelihood over output and datapoint dimension
        log_p_y_tgt = torch.sum(log_p_y_tgt, dim=(-2, -1))  # (n_tsk, n_ls, n_marg)
        log_p_y_ctx = torch.sum(log_p_y_ctx, dim=(-2, -1))  # (n_tsk, n_ls, n_marg)

        # compute normalized log-importance weights (works for empty context sets)
        #  for empty context sets this yields log_p_ctx = 0, i.e., p_ctx = 1 which is
        #  what we want in order to arrive at log_w_norm = log(1/S)
        mu_z_prior = mu_z_prior[:, None, :].expand(-1, n_ls, -1)
        cov_z_prior = cov_z_prior[:, None, :].expand(-1, n_ls, -1)
        log_w_norm = self._log_normalized_importance_weights_true_np(
            z=z,
            mu_z_prior=mu_z_prior,
            cov_z_prior=cov_z_prior,
            mu_z_posterior=mu_z_ctx,
            cov_z_posterior=cov_z_ctx,
            log_p_y_ctx=log_p_y_ctx,
        )

        # compute MC-average
        ll = torch.logsumexp(log_w_norm + log_p_y_tgt, dim=2)  # (n_tsk, n_ls)

        # sum task- and ls-dimensions
        ll = torch.sum(ll, dim=(0, 1))
        assert ll.ndim == 0

        # compute average log-likelihood over L * n_tgt with L = n_tsk * n_ls
        ll = ll / (n_tsk * n_ls * n_tgt)

        return ll

    def _log_normalized_importance_weights_true_np(
        self, z, mu_z_prior, cov_z_prior, mu_z_posterior, cov_z_posterior, log_p_y_ctx
    ):
        assert z.ndim == 4
        assert z.shape[-1] == self._config["d_z"]
        assert (
            mu_z_prior.ndim
            == cov_z_prior.ndim
            == mu_z_posterior.ndim
            == cov_z_posterior.ndim
            == 3
        )  # (n_tsk, n_ls, d_z)
        n_ls = z.shape[1]
        n_marg = z.shape[2]

        ## prior
        mu_z_prior = mu_z_prior[:, :, None, :].expand(-1, -1, n_marg, -1)
        cov_z_prior = cov_z_prior[:, :, None, :].expand(-1, -1, n_marg, -1)
        std_z_prior = torch.sqrt(cov_z_prior)
        gaussian_z_prior = torch.distributions.Normal(mu_z_prior, std_z_prior)
        log_p_prior = gaussian_z_prior.log_prob(z)  # (n_tsk, n_ls, n_marg, d_z)
        # factorized Gaussians -> joint log-prob = sum(log_probs)
        log_p_prior = torch.sum(log_p_prior, dim=-1)  # (n_tsk, n_ls, n_marg)

        ## posterior
        mu_z_posterior = mu_z_posterior[:, :, None, :].expand(-1, -1, n_marg, -1)
        cov_z_posterior = cov_z_posterior[:, :, None, :].expand(-1, -1, n_marg, -1)
        std_z_posterior = torch.sqrt(cov_z_posterior)
        gaussian_z_posterior = torch.distributions.Normal(
            mu_z_posterior, std_z_posterior
        )
        log_p_posterior = gaussian_z_posterior.log_prob(z)  # (n_tsk, n_ls, n_marg, d_z)
        # factorized Gaussians -> joint log-prob = sum(log_probs)
        log_p_posterior = torch.sum(log_p_posterior, dim=-1)  # (n_tsk, n_ls, n_marg)

        ## importance weights
        log_w = log_p_prior - log_p_posterior + log_p_y_ctx
        log_w_normalizer = torch.logsumexp(log_w, dim=2)  # (n_tsk, n_ls)
        log_w_normalizer = log_w_normalizer[:, :, None].expand(-1, -1, n_marg)
        log_w_norm = log_w - log_w_normalizer  # (n_tsk, n_ls, n_marg)

        return log_w_norm  # (n_tsk, n_ls, n_marg)

    def _elbo_np(
        self,
        x_tgt,
        y_tgt,
        mu_z_ctx,
        cov_z_ctx,
        agg_state_ctx,
        latent_obs_all,
    ):
        # computes the vi-inspired loss
        #  latent_obs_all are the latent observations w.r.t. context + target
        #  mu_z, cov_z, agg_state are the latent/agg states w.r.t. the *context set*

        # obtain shapes
        n_ls = mu_z_ctx.shape[1]
        n_tgt = x_tgt.shape[1]

        # compute posterior latent states w.r.t. the test sets
        mu_z_all = torch.zeros(mu_z_ctx.shape, device=self.device)
        cov_z_all = torch.zeros(cov_z_ctx.shape, device=self.device)
        for j in range(n_ls):
            cur_agg_state_old = tuple(entry[:, j, :] for entry in agg_state_ctx)
            cur_agg_state_new = self.aggregator.step(
                agg_state_old=cur_agg_state_old,
                latent_obs=latent_obs_all,
            )
            (cur_mu_z_all, cur_cov_z_all) = self.aggregator.agg2latent(
                cur_agg_state_new
            )
            mu_z_all[:, j, :] = cur_mu_z_all
            cov_z_all[:, j, :] = cur_cov_z_all

        # compute log likelihood using posterior latent states
        ll = self._conditional_ll(
            x_tgt=x_tgt, y_tgt=y_tgt, mu_z=mu_z_all, cov_z=cov_z_all
        )

        # compute kls between posteriors and corresponding priors
        std_z_ctx = torch.sqrt(cov_z_ctx)
        std_z_all = torch.sqrt(cov_z_all)
        gaussian_z_ctx = torch.distributions.Normal(loc=mu_z_ctx, scale=std_z_ctx)
        gaussian_z_tgt = torch.distributions.Normal(loc=mu_z_all, scale=std_z_all)
        kl = torch.distributions.kl.kl_divergence(gaussian_z_tgt, gaussian_z_ctx)
        # sum over latent dimension (diagonal Gaussians)
        kl = torch.sum(kl, axis=-1)
        # take mean over task and ls dimensions
        kl = torch.mean(kl, dim=[0, 1]).squeeze()

        # compute loss
        elbo = ll - kl / n_tgt

        return elbo

    def _predict(self, x, mu_z, cov_z, n_marg, return_latent_samples=False):
        assert x.ndim == 3  # (n_tsk, n_tst, d_x)
        assert mu_z.ndim == 3  # (n_tsk, n_ls, d_z)
        if cov_z is not None:
            assert mu_z.shape == cov_z.shape

        # collect shapes
        n_tsk = mu_z.shape[0]
        n_ls = mu_z.shape[1]
        d_z = mu_z.shape[2]

        if self.settings["loss_type"] == "PB":
            assert not return_latent_samples
            assert n_marg == 1
            if self.settings["aggregator_type"] == "BA":
                mu_y, std_y = self.decoder.decode(x, mu_z, cov_z)
                # add dummy n_marg-dim
                mu_y, std_y = mu_y[:, :, None, :, :], std_y[:, :, None, :, :]
            else:
                # add dummy n_marg-dim
                mu_z = mu_z[:, :, None, :]
                mu_z = mu_z.expand(n_tsk, n_ls, n_marg, d_z)
                mu_y, std_y = self.decoder.decode(x, mu_z)
        else:  # MC, IWMC, or VI losses:
            std_z = torch.sqrt(cov_z)

            # expand mu_z, std_z w.r.t. n_marg
            mu_z = mu_z[:, :, None, :]
            mu_z = mu_z.expand(n_tsk, n_ls, n_marg, d_z)
            std_z = std_z[:, :, None, :]
            std_z = std_z.expand(n_tsk, n_ls, n_marg, d_z)

            eps = self._rng.randn(*mu_z.shape)
            eps = torch.tensor(eps, dtype=torch.float32).to(self.device)
            z = mu_z + eps * std_z

            mu_y, std_y = self.decoder.decode(
                x=x, z=z, mu_z=mu_z, cov_z=cov_z
            )  # (n_tsk, n_ls, n_marg, n_tst, d_y)

        assert mu_y.ndim == 5 and std_y.ndim == 5

        if not return_latent_samples:
            # mu_y, std_y = (n_tsk, n_ls, n_marg, n_tst, d_y)
            return mu_y, std_y
        else:
            # mu_y, std_y = (n_tsk, n_ls, n_marg, n_tst, d_y)
            # z = (n_tsk, n_ls, n_marg, d_z)
            return mu_y, std_y, z

    def meta_train(
        self,
        benchmark_meta: MetaLearningBenchmark,
        n_tasks_train: int,
        validation_interval: Optional[int] = None,
        benchmark_val: Optional[MetaLearningBenchmark] = None,
        callback=None,
    ) -> float:
        def validate_now() -> bool:
            if validation_interval is None:
                return False

            # at beginning
            if self._n_meta_tasks_seen == 0:
                return True

            # if we jumped into the next validation_interval with the last batch
            if (
                (self._n_meta_tasks_seen - self._config["batch_size"])
                // validation_interval
                < self._n_meta_tasks_seen // validation_interval
            ):
                return True

            return False

        def validation_loss() -> float:
            if benchmark_val is None:
                return None

            with torch.no_grad():
                x_val, y_val = next(iter(dataloader_val))

                # normalize
                x_val, y_val = self._normalize_x(x_val), self._normalize_y(y_val)

                # compute validation loss
                loss = self._compute_loss(x=x_val, y=y_val, mode="val")
                loss = loss.cpu().numpy().item()

            return loss

        def optimizer_step() -> float:
            # perform optimizer step on batch of metadata
            x_meta, y_meta = next(iter(dataloader_meta))

            # drop last elements if batch is too large
            n_tasks_in_batch = x_meta.shape[0]
            if self._n_meta_tasks_seen + n_tasks_in_batch > n_tasks_train:
                n_tasks_remaining = n_tasks_train - self._n_meta_tasks_seen
                x_meta, y_meta = x_meta[:n_tasks_remaining], y_meta[:n_tasks_remaining]

            # normalize
            x_meta, y_meta = self._normalize_x(x_meta), self._normalize_y(y_meta)

            # reset gradient
            self._optimizer.zero_grad()

            # compute loss
            loss_meta = self._compute_loss(x=x_meta, y=y_meta, mode="meta")

            # perform gradient step on minibatch
            loss_meta.backward()
            self._optimizer.step()

            # update self.n_meta_tasks_seen
            n_tasks_in_batch = x_meta.shape[0]
            self._n_meta_tasks_seen += n_tasks_in_batch
            pbar.update(n_tasks_in_batch)

            return loss_meta.detach().numpy()

        # log
        self._logger.info(
            "Training model on {:d} tasks ({:d} remaining)...".format(
                n_tasks_train, max(n_tasks_train - self._n_meta_tasks_seen, 0)
            )
        )

        # determine normalizers on metadata before starting training
        # (normalizers could be already available if this fct was called with 0 iters)
        if self._n_meta_tasks_seen == 0 and not self._normalizers_available:
            self._determine_normalizers(benchmark=benchmark_meta)

        # set device for training
        self._set_device(self._config["device"])

        # create dataloader
        dataloader_meta = DataLoader(
            dataset=MetaLearningDataset(benchmark_meta),
            batch_size=self._config["batch_size"],
            collate_fn=lambda task_list: self._collate_batch(task_list),
        )
        if benchmark_val is not None:
            dataloader_val = DataLoader(
                dataset=MetaLearningDataset(benchmark_val),
                batch_size=self._config["batch_size"],
                collate_fn=lambda task_list: self._collate_batch(task_list),
            )
        else:
            dataloader_val = None

        # training loop
        loss_meta, loss_val = None, None
        with tqdm(
            total=n_tasks_train, leave=False, desc="meta-fit", mininterval=10
        ) as pbar:
            pbar.update(self._n_meta_tasks_seen)
            while self._n_meta_tasks_seen < n_tasks_train:
                if callback is not None:
                    callback(
                        n_meta_tasks_seen=self._n_meta_tasks_seen,
                        np_model=self,
                        metrics={"loss_meta": loss_meta}
                        if loss_meta is not None
                        else None,
                    )
                if validate_now():
                    loss_val = validation_loss()
                loss_meta = optimizer_step()
                pbar.set_postfix({"loss_meta": loss_meta, "loss_val": loss_val})

            # compute loss_val once again at the end
            loss_val = validation_loss()

        self._set_device("cpu")
        self._logger.info("Training finished successfully!")

        return loss_val

    @torch.no_grad()
    def predict(self, x: np.ndarray, n_samples: int = 1) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        # check input data
        self._check_data_shapes(x=x)

        # prepare x
        has_tsk_dim = x.ndim == 3
        x = self._prepare_data_for_testing(x)
        x = self._normalize_x(x)

        # read out last latent state
        ls = self.aggregator.last_latent_state
        mu_z = ls[0][:, None, :]
        cov_z = ls[1][:, None, :] if ls[1] is not None else None

        # obtain predictions
        mu_y, std_y = self._predict(x, mu_z, cov_z, n_marg=n_samples)

        # denormalize the predictions
        mu_y = self._denormalize_mu_y(mu_y)  # (n_tsk, n_ls, n_marg, n_tst, d_y)
        if (
            self._config["decoder_kwargs"]["input_mlp_std_y"] == ""
            and self._config["decoder_kwargs"]["global_std_y_is_learnable"]
        ):
            raise NotImplementedError  # think about denormalization
        if not self._config["decoder_kwargs"]["input_mlp_std_y"] == "":
            std_y = self._denormalize_std_y(std_y)  # (n_tsk, n_ls, n_marg, n_tst, d_y)

        # check that target and context data are consistent
        if has_tsk_dim and mu_y.shape[0] != x.shape[0]:
            raise NotImplementedError(
                "Target and context data have different numbers of tasks!"
            )

        # squeeze latent state dimension (this is always singleton here)
        mu_y, std_y = mu_y.squeeze(1), std_y.squeeze(1)  # squeeze n_ls dimension

        # squeeze marginalization dimension (this is always singleton here)
        if n_samples == 1:
            mu_y, std_y = mu_y.squeeze(1), std_y.squeeze(1)  # squeeze n_marg dimension

        # squeeze task dimension (if singleton)
        if not has_tsk_dim:
            mu_y = mu_y.squeeze(0)
            std_y = std_y.squeeze(0)
        mu_y, std_y = mu_y.numpy(), std_y.numpy()

        return mu_y, std_y ** 2  # ([n_tsk,], [n_samples], n_pts, d_y)

    @torch.no_grad()
    def predict_importance_weights(
        self, x: np.ndarray, task_ctx: MetaLearningTask, n_marg: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert not self._config["predictions_are_deterministic"]

        # prepare x
        has_tsk_dim = x.ndim == 3
        x = self._prepare_data_for_testing(x)
        x = self._normalize_x(x)
        if x.shape[0] > 1:
            raise NotImplementedError  # not implemented for more than one task

        # check input data
        x_ctx, y_ctx = task_ctx.x, task_ctx.y
        self._check_data_shapes(x=x)
        self._check_data_shapes(x=x_ctx, y=y_ctx)

        # prepare x_ctx and y_ctx
        x_ctx = self._prepare_data_for_testing(x_ctx)
        y_ctx = self._prepare_data_for_testing(y_ctx)
        x_ctx = self._normalize_x(x_ctx)
        y_ctx = self._normalize_y(y_ctx)

        # read out prior latent state
        ls_prior = self.aggregator.initial_latent_state
        mu_z_prior = ls_prior[0][:, None, :]
        cov_z_prior = ls_prior[1][:, None, :]

        # read out posterior latent state
        ls_posterior = self.aggregator.last_latent_state
        mu_z_posterior = ls_posterior[0][:, None, :]
        cov_z_posterior = ls_posterior[1][:, None, :]

        # shapes
        n_tgt = x.shape[1]
        n_ls = mu_z_posterior.shape[1]

        # obtain predictions (use same sample set for both tgt and ctx)
        all_x = torch.cat((x, x_ctx), dim=1)
        all_mu_y, all_std_y, z = self._predict(
            all_x,
            mu_z_posterior,
            cov_z_posterior,
            n_marg=n_marg,
            return_latent_samples=True,
        )
        # all_mu_y, all_std_y shape = (n_tsk, n_ls, n_marg, n_tst, d_y)
        # z shape = (n_tsk, n_ls, n_marg, d_z)
        mu_y, mu_y_ctx = all_mu_y[:, :, :, :n_tgt, :], all_mu_y[:, :, :, n_tgt:, :]
        std_y, std_y_ctx = (
            all_std_y[:, :, :, :n_tgt, :],
            all_std_y[:, :, :, n_tgt:, :],
        )

        # add latent state and marginalization dimension to y-values
        y_ctx = y_ctx[:, None, None, :, :].expand(-1, n_ls, n_marg, -1, -1)

        # compute log-likelihood of context datapoints
        gaussian_ctx = torch.distributions.Normal(mu_y_ctx, std_y_ctx)
        log_p_ctx = gaussian_ctx.log_prob(y_ctx)  # (n_tsk, n_ls, n_marg, n_ctx, d_y)

        # sum log-likelihood of context datapoints over output and datapoint dimension
        #  for empty context sets this yields log_p_ctx = 0, i.e., p_ctx = 1 which is
        #  what we want in order to arrive at log_w_norm = log(1/S)
        log_p_ctx = torch.sum(log_p_ctx, dim=(-2, -1))  # (n_tsk, n_ls, n_marg)

        # compute normalized log-importance weights
        log_w_norm = self._log_normalized_importance_weights_true_np(
            z=z,
            mu_z_prior=mu_z_prior,
            cov_z_prior=cov_z_prior,
            mu_z_posterior=mu_z_posterior,
            cov_z_posterior=cov_z_posterior,
            log_p_y_ctx=log_p_ctx,
        )  # (n_tsk, n_ls, n_marg)

        # denormalize the predictions
        mu_y = self._denormalize_mu_y(mu_y)  # (n_tsk, n_ls, n_marg, n_tst, d_y)
        std_y = self._denormalize_std_y(std_y)  # (n_tsk, n_ls, n_marg, n_tst, d_y)

        # check that target and context data are consistent
        if has_tsk_dim and mu_y.shape[0] != x.shape[0]:
            raise NotImplementedError(
                "Target and context data have different numbers of tasks!"
            )

        # squeeze latent state dimension (this is always singleton here)
        mu_y, std_y = mu_y.squeeze(1), std_y.squeeze(1)
        log_w_norm = log_w_norm.squeeze(1)

        # squeeze task dimension (if singleton)
        if not has_tsk_dim:
            mu_y = mu_y.squeeze(0)
            std_y = std_y.squeeze(0)
            log_w_norm = log_w_norm.squeeze(0)
        mu_y, std_y = mu_y.numpy(), std_y.numpy()
        log_w_norm = log_w_norm.numpy()

        # shapes:
        #  mu_y, std_y ** 2: ([n_tsk,], n_marg, n_pts, d_y)
        #  log_w_norm: ([n_tsk], n_marg)
        return mu_y, std_y ** 2, log_w_norm

    @torch.no_grad()
    def adapt(self, x: np.ndarray, y: np.ndarray) -> None:
        self._check_data_shapes(x=x, y=y)

        # prepare x and y
        x = self._prepare_data_for_testing(x)
        y = self._prepare_data_for_testing(y)
        x = self._normalize_x(x)
        y = self._normalize_y(y)

        # accumulate data in aggregator
        self.aggregator.reset(n_tsk=x.shape[0])
        if x.shape[1] > 0:
            latent_obs = self.encoder.encode(x=x, y=y)
            self.aggregator.update(latent_obs)

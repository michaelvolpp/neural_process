import torch

from np_util.mlp import MLP, LearnableTensor
from np_util.output_trafo import output_trafo, inv_output_trafo


# TODO: decoder networks shall inherit from a proper torch module


class DecoderNetworkPB:
    def __init__(self, **kwargs):
        self.d_x = kwargs["d_x"]
        self.d_y = kwargs["d_y"]
        self.d_z = kwargs["d_z"]
        self.arch = kwargs["arch"]

        # process network shapes
        self.mlp_layers_mu_y = kwargs["mlp_layers_mu_y"]
        self.mlp_layers_std_y = kwargs["mlp_layers_std_y"]

        self.f_act = kwargs["f_act"]
        self.safe_log = 1e-8
        self.seed = kwargs["seed"]

        # TODO merge this with self.arch
        assert kwargs["input_mlp_std_y"] == "xz"

        self.mu_y_net = self.std_y_net = self.mu_y_std_y_net = None
        self.create_networks()

    def to(self, device):
        if (
            self.arch == "separate_networks_combined_input"
            or self.arch == "separate_networks_separate_input"
        ):
            self.mu_y_net = self.mu_y_net.to(device)
            self.std_y_net = self.std_y_net.to(device)
        elif self.arch == "two_heads":
            self.mu_y_std_y_net = self.mu_y_std_y_net.to(device)

    def save_weights(self, logpath, epoch):
        if (
            self.arch == "separate_networks_combined_input"
            or self.arch == "separate_networks_separate_input"
        ):
            self.mu_y_net.save_weights(path=logpath, epoch=epoch)
            self.std_y_net.save_weights(path=logpath, epoch=epoch)
        elif self.arch == "two_heads":
            self.mu_y_std_y_net.save_weights(path=logpath, epoch=epoch)

    def load_weights(self, logpath, epoch):
        if (
            self.arch == "separate_networks_combined_input"
            or self.arch == "separate_networks_separate_input"
        ):
            self.mu_y_net.load_weights(path=logpath, epoch=epoch)
            self.std_y_net.load_weights(path=logpath, epoch=epoch)

    def create_networks(self):
        if self.arch == "separate_networks_combined_input":
            self.mu_y_net = MLP(
                name="decoder_mu",
                d_in=self.d_x + self.d_z * 2,
                d_out=self.d_y,
                mlp_layers=self.mlp_layers_mu_y,
                f_act=self.f_act,
                seed=self.seed,
            )
            self.std_y_net = MLP(
                name="decoder_cov",
                d_in=self.d_x + self.d_z * 2,
                d_out=self.d_y,
                mlp_layers=self.mlp_layers_std_y,
                f_act=self.f_act,
                seed=self.seed,
            )
        elif self.arch == "separate_networks_separate_input":
            self.mu_y_net = MLP(
                name="decoder_mu",
                d_in=self.d_x + self.d_z,
                d_out=self.d_y,
                mlp_layers=self.mlp_layers_mu_y,
                f_act=self.f_act,
                seed=self.seed,
            )
            self.std_y_net = MLP(
                name="decoder_cov",
                d_in=self.d_x + self.d_z,
                d_out=self.d_y,
                mlp_layers=self.mlp_layers_std_y,
                f_act=self.f_act,
                seed=self.seed,
            )
        elif self.arch == "two_heads":
            self.mu_y_std_y_net = MLP(
                name="decoder_mu_cov",
                d_in=self.d_x + self.d_z * 2,
                d_out=2 * self.d_y,
                mlp_layers=self.mlp_layers_mu_y,  # ignore self.mlp_layers_std_y
                f_act=self.f_act,
                seed=self.seed,
            )
        else:
            raise ValueError("Unknown network type: {}!".format(self.arch))

    def decode(self, x, mu_z, cov_z):
        assert x.ndim == 3
        assert mu_z.ndim == cov_z.ndim == 3

        n_tsk = mu_z.shape[0]
        n_ls = mu_z.shape[1]
        n_tst = x.shape[1]

        # covariance parametrization: work on log(cov) as it is better scaled!?
        cov_z = self.parametrize_latent_cov(cov_z)

        # add latent-state-wise batch dimension to X
        x = x[:, None, :, :]
        x = x.expand(n_tsk, n_ls, n_tst, self.d_x)

        # add dataset-wise batch dimension to latent states
        mu_z = mu_z[:, :, None, :]
        cov_z = cov_z[:, :, None, :]

        # prepare input to decoder network
        if self.arch == "separate_networks_combined_input":
            mu_z_cov_z = torch.cat((mu_z, cov_z), dim=3)
            mu_z_cov_z = mu_z_cov_z.expand((n_tsk, n_ls, n_tst, self.d_z * 2))
            input_mu = input_std = torch.cat((x, mu_z_cov_z), dim=3)
        elif self.arch == "separate_networks_separate_input":
            mu_z = mu_z.expand((n_tsk, n_ls, n_tst, self.d_z))
            cov_z = cov_z.expand((n_tsk, n_ls, n_tst, self.d_z))
            input_mu = torch.cat((x, mu_z), dim=3)
            input_std = torch.cat((x, cov_z), dim=3)
        elif self.arch == "two_heads":
            mu_z_cov_z = torch.cat((mu_z, cov_z), dim=3)
            mu_z_cov_z = mu_z_cov_z.expand((n_tsk, n_ls, n_tst, self.d_z * 2))
            input_two_head = torch.cat((x, mu_z_cov_z), dim=3)

        # decode
        if (
            self.arch == "separate_networks_combined_input"
            or self.arch == "separate_networks_separate_input"
        ):
            mu_y = self.mu_y_net(input_mu)
            std_y = self.std_y_net(input_std)
        elif self.arch == "two_heads":
            # TODO: fix this
            mu_y_std_y = self.mu_y_std_y_net(input_two_head)
            mu_y = mu_y_std_y[:, : self.d_y]
            std_y = mu_y_std_y[:, self.d_y :]

        # deparametrize
        std_y = output_trafo(std_y, lower_bound=1e-4)

        return mu_y, std_y

    def parametrize_latent_cov(self, cov):
        cov = cov + self.safe_log
        parametrized_cov = torch.log(cov)

        return parametrized_cov

    @property
    def parameters(self):
        params = []
        if (
            self.arch == "separate_networks_combined_input"
            or self.arch == "separate_networks_separate_input"
        ):
            params += list(self.mu_y_net.parameters()) + list(
                self.std_y_net.parameters()
            )
        elif self.arch == "two_heads":
            params += list(self.mu_y_std_y_net.parameters())
        return params


class DecoderNetworkSamples:
    def __init__(self, **kwargs):
        self.d_x = kwargs["d_x"]
        self.d_y = kwargs["d_y"]
        self.d_z = kwargs["d_z"]
        self.arch = kwargs["arch"]

        self.mlp_layers_mu_y = kwargs["mlp_layers_mu_y"]
        self.f_act = kwargs["f_act"]
        self.seed = kwargs["seed"]

        # process std_y_network/global_std_y
        self.input_mlp_std_y = kwargs["input_mlp_std_y"]
        self.mlp_layers_std_y = kwargs.get("mlp_layers_std_y", None)
        global_std_y_is_learnable = kwargs.get("global_std_y_is_learnable", None)
        global_std_y_init = kwargs.get("global_std_y", None)
        # check arguments are valid
        assert self.input_mlp_std_y in [
            "",
            "x",
            "z",
            "xz",
            "cov_z",
        ], "Invalid output variance dependence specified."
        if self.arch != "separate_networks":
            assert (
                self.input_mlp_std_y == "xz"
            ), "Joint networks require input_mlp_std_y = 'xz'. {} not allowed!".format(
                self.input_mlp_std_y
            )
        if self.input_mlp_std_y != "":
            assert self.mlp_layers_std_y is not None
            assert global_std_y_is_learnable is None
            assert global_std_y_init is None

            self.learnable_std_y = True
        else:
            assert self.mlp_layers_std_y is None
            assert global_std_y_init is not None
            assert global_std_y_is_learnable is not None

            global_std_y_init = torch.tensor(global_std_y_init)
            if global_std_y_init.ndim == 0:
                global_std_y_init = global_std_y_init.expand(self.d_y)
            elif global_std_y_init.ndim == 1:
                assert global_std_y_init.nelement == self.d_y
            else:
                raise ValueError("std_y must be scalar or vector.")

            # apply inverse output transform
            self._global_std_y = inv_output_trafo(global_std_y_init, lower_bound=1e-4)
            self.learnable_std_y = global_std_y_is_learnable

        self.mu_y_net = self.std_y_net = self.mu_y_std_y_net = None
        self.create_networks()

    @property
    def global_std_y(self):
        """Converts internal parametrization to std_y."""
        if self.input_mlp_std_y != "":
            return None
        if self.learnable_std_y:
            return output_trafo(self.std_y_net.value, lower_bound=1e-4)
        else:
            return output_trafo(self._global_std_y, lower_bound=1e-4)

    def to(self, device):
        if self.arch == "separate_networks":
            self.mu_y_net = self.mu_y_net.to(device)
            if self.learnable_std_y:  # MLP or LearnableTensor
                self.std_y_net = self.std_y_net.to(device)
            else:  # just a torch tensor, as no weight saving is needed
                self._global_std_y = self._global_std_y.to(device)
        elif self.arch == "two_heads":
            self.mu_y_std_y_net = self.mu_y_std_y_net.to(device)

    def save_weights(self, logpath, epoch):
        if self.arch == "separate_networks":
            self.mu_y_net.save_weights(path=logpath, epoch=epoch)
            if self.learnable_std_y:
                self.std_y_net.save_weights(path=logpath, epoch=epoch)
        elif self.arch == "two_heads":
            self.mu_y_std_y_net.save_weights(path=logpath, epoch=epoch)

    def load_weights(self, logpath, epoch):
        if self.arch == "separate_networks":
            self.mu_y_net.load_weights(path=logpath, epoch=epoch)
            if self.learnable_std_y:
                self.std_y_net.load_weights(path=logpath, epoch=epoch)
        elif self.arch == "two_heads":
            self.mu_y_std_y_net.load_weights(path=logpath, epoch=epoch)

    def create_networks(self):
        if self.input_mlp_std_y == "xz":
            d_in_std_y = self.d_x + self.d_z
        elif self.input_mlp_std_y == "x":
            d_in_std_y = self.d_x
        elif self.input_mlp_std_y == "z" or self.input_mlp_std_y == "cov_z":
            d_in_std_y = self.d_z
        else:
            d_in_std_y = None  # constant std_y
        if self.arch == "separate_networks":
            self.mu_y_net = MLP(
                name="decoder_mu",
                d_in=self.d_x + self.d_z,
                d_out=self.d_y,
                mlp_layers=self.mlp_layers_mu_y,
                f_act=self.f_act,
                seed=self.seed,
            )
            if self.learnable_std_y:
                if self.input_mlp_std_y != "":
                    self.std_y_net = MLP(
                        name="decoder_cov",
                        d_in=d_in_std_y,
                        d_out=self.d_y,
                        mlp_layers=self.mlp_layers_std_y,
                        f_act=self.f_act,
                        seed=self.seed,
                    )
                else:
                    self.std_y_net = LearnableTensor(
                        name="decoder_cov",
                        init_mean=self._global_std_y,
                        uniform_init_delta=None,
                        # uniform_init_delta=0.1 * self._global_std_y, # in softplus space
                        # seed=self.seed,
                    )
        elif self.arch == "two_heads":
            self.mu_y_std_y_net = MLP(
                name="decoder_mu_cov",
                d_in=self.d_x + self.d_z,
                d_out=2 * self.d_y,
                mlp_layers=self.mlp_layers_mu_y,  # ignore mlp_layers_std_y
                f_act=self.f_act,
                seed=self.seed,
            )
        else:
            raise ValueError("Unknown decoder network arch!")

    def decode(self, x, z, mu_z=None, cov_z=None):
        mu_z = None  # unused
        if self.input_mlp_std_y == "cov_z":
            assert cov_z is not None
        else:
            cov_z = None  # unused

        assert x.ndim == 3  # (n_tsk, n_tst, d_x)
        assert z.ndim == 4  # (n_tsk, n_ls, n_marg, d_z)
        if cov_z is not None:
            assert cov_z.ndim == 3  # (n_tsk, n_ls, d_z)

        n_tsk = z.shape[0]
        n_ls = z.shape[1]
        n_marg = z.shape[2]
        n_tst = x.shape[1]

        # add latent-state-wise batch dimension to x
        x = x[:, None, None, :, :]
        x = x.expand(n_tsk, n_ls, n_marg, n_tst, self.d_x)

        # add dataset-wise batch dimension to sample
        z = z[:, :, :, None, :]
        z = z.expand((n_tsk, n_ls, n_marg, n_tst, self.d_z))

        # add dataset-wise and marginalization batch dimension to cov_z
        if cov_z is not None:
            cov_z = cov_z[:, :, None, None, :]
            cov_z = cov_z.expand((n_tsk, n_ls, n_marg, n_tst, self.d_z))

        # decode
        # prepare input to decoder network
        input_mu = torch.cat((x, z), dim=4)
        if self.arch == "separate_networks":
            if self.input_mlp_std_y != "":
                if self.input_mlp_std_y == "xz":
                    input_std = [x, z]
                elif self.input_mlp_std_y == "x":
                    input_std = [x]
                elif self.input_mlp_std_y == "z":
                    input_std = [z]
                elif self.input_mlp_std_y == "cov_z":
                    input_std = [cov_z]
                input_std = torch.cat(input_std, dim=4)

            mu_y = self.mu_y_net(input_mu)
            if self.learnable_std_y:
                std_y = (
                    self.std_y_net(input_std)
                    if self.input_mlp_std_y != ""
                    else self.std_y_net(batch_shape=x.shape[:-1])
                )
            else:
                std_y = self._global_std_y[None, None, None, None, :].expand(
                    x.shape[:-1] + (self.d_y,)
                )
        elif self.arch == "two_heads":
            mu_cov_y = self.mu_y_std_y_net(input_mu)  # full input
            mu_y = mu_cov_y[:, : self.d_y]
            std_y = mu_cov_y[:, self.d_y :]

        # deparametrize
        std_y = output_trafo(std_y, lower_bound=1e-4)

        return mu_y, std_y

    @property
    def parameters(self):
        params = []
        if self.arch == "separate_networks":
            params += list(self.mu_y_net.parameters())
            if self.learnable_std_y:
                params += list(self.std_y_net.parameters())
        elif self.arch == "two_heads":
            params += list(self.mu_y_std_y_net.parameters())
        return params

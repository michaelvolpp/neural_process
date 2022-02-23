import torch

from np_util.mlp import MLP, LearnableTensor
from np_util.output_trafo import output_trafo, inv_output_trafo


class BayesianAggregator:
    def __init__(self, **kwargs):
        self.d_z = self.d_r = kwargs["d_z"]
        self.mu_z_0 = torch.zeros((self.d_z,))

        ## deal with var_z_0
        # create var_z_0
        var_z_0 = kwargs.get("var_z_0", 1.0)
        var_z_0 = torch.tensor(var_z_0)
        if var_z_0.ndim == 0:
            var_z_0 = var_z_0.expand(self.d_z)
        elif var_z_0.ndim == 1:
            assert var_z_0.nelement == self.d_z
        else:
            raise ValueError("var_z_0 has wrong shape!")
        # go to parametrized space
        self.var_z_0_parametrized = inv_output_trafo(var_z_0, lower_bound=1e-4)
        # create LearnableTensor, if required
        self.var_z_0_is_learnable = kwargs.get("var_z_0_is_learnable", False)
        if self.var_z_0_is_learnable:
            self.var_z_0_parametrized = LearnableTensor(
                name="var_z_0_parametrized",
                init_mean=self.var_z_0_parametrized,
                uniform_init_delta=None,
                # uniform_init_delta=0.1 * self.var_z_0_parametrized,
                # seed=kwargs["seed"],
            )

        self.mu_z = None
        self.var_z = None

        self.n_tsk = None

        self.device = None

    @property
    def agg_state(self):
        return self.mu_z, self.var_z

    @property
    def initial_agg_state(self):
        # deparametrize var_z_0
        var_z_0_parametrized = (
            self.var_z_0_parametrized
            if not self.var_z_0_is_learnable
            else self.var_z_0_parametrized.value
        )
        var_z_0 = output_trafo(var_z_0_parametrized, lower_bound=1e-4)

        # add batch dimension
        mu_z_0 = self.mu_z_0[None, :].expand(self.n_tsk, self.d_z)
        var_z_0 = var_z_0[None, :].expand(self.n_tsk, self.d_z)

        return mu_z_0, var_z_0

    @property
    def last_agg_state(self):
        return self.mu_z[:, -1, :], self.var_z[:, -1, :]

    @property
    def latent_state(self):
        return self.agg2latent(self.agg_state)

    @property
    def initial_latent_state(self):
        return self.agg2latent(self.initial_agg_state)

    @property
    def last_latent_state(self):
        return self.agg2latent(self.last_agg_state)

    @property
    def parameters(self):
        params = []
        if self.var_z_0_is_learnable:
            params += list(self.var_z_0_parametrized.parameters())
        return params

    def save_weights(self, logpath, epoch):
        if self.var_z_0_is_learnable:
            self.var_z_0_parametrized.save_weights(path=logpath, epoch=epoch)

    def load_weights(self, logpath, epoch):
        if self.var_z_0_is_learnable:
            self.var_z_0_parametrized.load_weights(path=logpath, epoch=epoch)

    def delete_all_weight_files(self, logpath):
        if self.var_z_0_is_learnable:
            self.var_z_0_parametrized.delete_all_weight_files(path=logpath)

    def to(self, device):
        self.device = device

    def _append_agg_state(self, agg_state):
        mu_z = agg_state[0]
        var_z = agg_state[1]

        assert mu_z.ndim == var_z.ndim == 2
        assert mu_z.shape == var_z.shape

        # add ls dimension
        mu_z = mu_z[:, None, :]
        var_z = var_z[:, None, :]

        # append state
        self.mu_z = torch.cat((self.mu_z, mu_z), dim=1)
        self.var_z = torch.cat((self.var_z, var_z), dim=1)

    def reset(self, n_tsk):
        self.n_tsk = n_tsk

        # send initial state to device
        # TODO: check this!
        self.mu_z_0 = self.mu_z_0.to(self.device)
        self.var_z_0_parametrized = self.var_z_0_parametrized.to(self.device)

        # expand task dimension
        mu_z = self.mu_z_0[None, None, :].expand(self.n_tsk, -1, -1)
        var_z_0_parametrized = (
            self.var_z_0_parametrized
            if not self.var_z_0_is_learnable
            else self.var_z_0_parametrized.value
        )
        var_z_0_parametrized = var_z_0_parametrized[None, None, :].expand(
            self.n_tsk, -1, -1
        )
        var_z = output_trafo(
            var_z_0_parametrized,
            lower_bound=1e-4,
        )

        # send to device
        self.mu_z = mu_z.clone().to(self.device)
        self.var_z = var_z.clone().to(self.device)

    @staticmethod
    def _decode_step_input(latent_obs, agg_state_old):
        assert len(latent_obs) == 2
        r = latent_obs[0]
        var_r = latent_obs[1]
        mu_z = agg_state_old[0]
        var_z = agg_state_old[1]
        return r, var_r, mu_z, var_z

    def agg2latent(self, agg_state):
        return agg_state[0], agg_state[1]

    def sequential_step(self, latent_obs, agg_state_old):
        # decode input
        r, var_r, mu_z, var_z = self._decode_step_input(
            latent_obs=latent_obs, agg_state_old=agg_state_old
        )

        # check input
        assert r.ndim == var_r.ndim == 2
        assert r.shape == var_r.shape
        assert mu_z.ndim == var_z.ndim == 2
        assert mu_z.shape == var_z.shape
        assert r.shape[0] == mu_z.shape[0]
        assert r.shape[1] == mu_z.shape[1]

        S = var_z + var_r
        S_inv = 1 / S
        K = var_z * S_inv
        v = r - mu_z
        mu_z_new = mu_z + K * v
        cov_z_new = var_z - K * var_z

        return mu_z_new, cov_z_new

    def step(self, latent_obs, agg_state_old):
        # decode input
        r, var_r, mu_z, var_z = self._decode_step_input(
            latent_obs=latent_obs, agg_state_old=agg_state_old
        )

        # check input
        assert r.ndim == var_r.ndim == 3
        assert r.shape == var_r.shape
        assert mu_z.ndim == var_z.ndim == 2
        assert mu_z.shape == var_z.shape
        assert r.shape[0] == mu_z.shape[0]
        assert r.shape[2] == mu_z.shape[1]

        if r.shape[1] == 0:
            return mu_z, var_z  # nothing to do

        if r.shape[1] == 1:
            # for one-point updates the sequential version is faster
            return self.sequential_step(
                latent_obs=(r[:, 0, :], var_r[:, 0, :]), agg_state_old=(mu_z, var_z)
            )

        v = r - mu_z[:, None, :]
        cov_w_inv = 1 / var_r
        cov_z_new = 1 / (1 / var_z + torch.sum(cov_w_inv, dim=1))
        mu_z_new = mu_z + cov_z_new * torch.sum(cov_w_inv * v, dim=1)

        return mu_z_new, cov_z_new

    def update_seq(self, latent_obs):
        raise NotImplementedError
        # perform update
        new_agg_state = self.sequential_step(
            latent_obs=latent_obs, agg_state_old=self.last_agg_state
        )

        # append new state
        self._append_agg_state(agg_state=new_agg_state)

    def update(self, latent_obs, incremental=False):
        r = latent_obs[0]
        if r.shape[1] == 0:
            return  # nothing to do

        # perform update
        if incremental:  # incremental update is not applicable with self-attention!
            agg_state_old = self.last_agg_state
        else:
            agg_state_old = self.initial_agg_state
        new_agg_state = self.step(latent_obs=latent_obs, agg_state_old=agg_state_old)

        # append new state
        self._append_agg_state(new_agg_state)


class MeanAggregator:
    def __init__(self, **kwargs):
        self.d_r = kwargs["d_r"]
        self.d_z = None
        # value does not matter as it appears only multiplied by self.n_agg_init_scalar
        self.mu_r_init_scalar = 0.0
        self.n_agg_init_scalar = 0

        self.mu_r_init = None
        self.n_agg_init = None
        self.mu_r = None  # the current mean
        self.n_agg = None  # the number of datapoints aggregated

        self.n_tsk = None  # number of tasks

        self.device = None

    @property
    def latent_state(self):
        return self.agg2latent(self.agg_state)

    @property
    def initial_latent_state(self):
        return self.agg2latent(self.initial_agg_state)

    @property
    def last_latent_state(self):
        return self.agg2latent(self.last_agg_state)

    @property
    def agg_state(self):
        return self.mu_r, self.n_agg

    @property
    def initial_agg_state(self):
        return self.mu_r_init[:, 0, :], self.n_agg_init[:, 0, :]

    @property
    def last_agg_state(self):
        return self.mu_r[:, -1, :], self.n_agg[:, -1, :]

    @property
    def parameters(self):
        return []

    def save_weights(self, logpath, epoch):
        pass

    def load_weights(self, logpath, epoch):
        pass

    def delete_all_weight_files(self, logpath):
        pass

    def to(self, device):
        self.device = device

    def _append_agg_state(self, agg_state):
        mu_r = agg_state[0]
        n_agg = agg_state[1]

        assert mu_r.ndim == 2

        # add ls dimension
        mu_r = mu_r[:, None, :]
        n_agg = n_agg[:, None, :]

        # append state
        self.mu_r = torch.cat((self.mu_r, mu_r), dim=1)
        self.n_agg = torch.cat((self.n_agg, n_agg), dim=1)

    def reset(self, n_tsk):
        self.n_tsk = n_tsk

        # create intial state
        mu_r = torch.ones(self.d_r) * self.mu_r_init_scalar
        n_agg = torch.ones(1) * self.n_agg_init_scalar

        # add task and states dimensions
        mu_r = mu_r[None, None, :]
        n_agg = n_agg[None, None, :]

        # expand task dimension
        mu_r = mu_r.expand(self.n_tsk, -1, -1)
        n_agg = n_agg.expand(self.n_tsk, -1, -1)

        # send to device and set attributes
        self.mu_r = mu_r.clone().to(self.device)
        self.mu_r_init = mu_r.clone().to(self.device)
        self.n_agg = n_agg.clone().to(self.device)
        self.n_agg_init = n_agg.clone().to(self.device)

    def agg2latent(self, agg_state):
        return agg_state[0], None

    def _decode_step_input(self, latent_obs, agg_state_old):
        assert len(latent_obs) == 1
        r = latent_obs[0]
        n_r = torch.tensor(r.shape[1], dtype=torch.float).to(self.device)
        mu_r = agg_state_old[0]
        n_agg = agg_state_old[1]
        return r, n_r, mu_r, n_agg

    def step(self, latent_obs, agg_state_old):
        # decode input
        r, n_r, mu_r, n_agg = self._decode_step_input(
            latent_obs=latent_obs, agg_state_old=agg_state_old
        )

        # check input
        assert r.ndim == 3
        assert mu_r.ndim == 2
        assert r.shape[0] == mu_r.shape[0]
        assert r.shape[2] == mu_r.shape[1]
        assert r.shape[1] > 0

        mu_r_new = 1 / (n_r + n_agg) * (n_agg * mu_r + n_r * torch.mean(r, dim=1))
        n_agg_new = n_agg + n_r

        return mu_r_new, n_agg_new

    def update(self, latent_obs, incremental=False):
        r = latent_obs[0]
        if r.shape[1] == 0:
            return  # nothing to do

        # perform update
        if incremental:  # incremental update is not applicable with self-attention!
            agg_state_old = self.last_agg_state
        else:
            agg_state_old = self.initial_agg_state
        new_agg_state = self.step(latent_obs=latent_obs, agg_state_old=agg_state_old)

        # append new state
        self._append_agg_state(new_agg_state)


class MeanAggregatorRtoZ(MeanAggregator):
    def __init__(self, **kwargs):
        # TODO: implement two-headed r-to-z networks
        super().__init__(**kwargs)
        self.d_z = kwargs["d_z"]
        self.mean_z_mlp = MLP(
            name="agg2mean_z",
            d_in=kwargs["d_r"],
            d_out=kwargs["d_z"],
            mlp_layers=kwargs["mlp_layers_r_to_mu_z"],
            f_act=kwargs["f_act"],
            f_out=(None, {}),
            seed=kwargs["seed"],
        )
        self.cov_z_mlp = MLP(
            name="agg2cov_z",
            d_in=kwargs["d_r"],
            d_out=kwargs["d_z"],
            mlp_layers=kwargs["mlp_layers_r_to_cov_z"],
            f_act=kwargs["f_act"],
            f_out=(None, {}),
            seed=kwargs["seed"],
        )

    def agg2latent(self, agg_state):
        return (
            self.mean_z_mlp(agg_state[0]),
            output_trafo(self.cov_z_mlp(agg_state[0]), lower_bound=1e-4),
        )

    def to(self, device):
        self.device = device
        self.mean_z_mlp.to(device)
        self.cov_z_mlp.to(device)

    @property
    def parameters(self):
        return list(self.mean_z_mlp.parameters()) + list(self.cov_z_mlp.parameters())

    def save_weights(self, logpath, epoch):
        self.mean_z_mlp.save_weights(path=logpath, epoch=epoch)
        self.cov_z_mlp.save_weights(path=logpath, epoch=epoch)

    def load_weights(self, logpath, epoch):
        self.mean_z_mlp.load_weights(path=logpath, epoch=epoch)
        self.cov_z_mlp.load_weights(path=logpath, epoch=epoch)

    def delete_all_weight_files(self, logpath):
        self.mean_z_mlp.delete_all_weight_files(path=logpath)
        self.cov_z_mlp.delete_all_weight_files(path=logpath)


class MaxAggregator:
    def __init__(self, **kwargs):
        self.d_r = kwargs["d_r"]
        self.d_z = None
        self.max_r_init_scalar = 0.0

        self.max_r_init = None
        self.max_r = None  # the current mean

        self.n_tsk = None  # number of tasks

        self.device = None

    @property
    def latent_state(self):
        return self.agg2latent(self.agg_state)

    @property
    def initial_latent_state(self):
        return self.agg2latent(self.initial_agg_state)

    @property
    def last_latent_state(self):
        return self.agg2latent(self.last_agg_state)

    @property
    def agg_state(self):
        return (self.max_r,)

    @property
    def initial_agg_state(self):
        return (self.max_r_init[:, 0, :],)

    @property
    def last_agg_state(self):
        return (self.max_r[:, -1, :],)

    @property
    def parameters(self):
        return []

    def save_weights(self, logpath, epoch):
        pass

    def load_weights(self, logpath, epoch):
        pass

    def delete_all_weight_files(self, logpath):
        pass

    def to(self, device):
        self.device = device

    def _append_agg_state(self, agg_state):
        max_r = agg_state[0]

        assert max_r.ndim == 2

        # add ls dimension
        max_r = max_r[:, None, :]

        # append state
        self.max_r = torch.cat((self.max_r, max_r), dim=1)

    def reset(self, n_tsk):
        self.n_tsk = n_tsk

        # create intial state
        max_r = torch.ones(self.d_r) * self.max_r_init_scalar

        # add task and states dimensions
        max_r = max_r[None, None, :]

        # expand task dimension
        max_r = max_r.expand(self.n_tsk, -1, -1)

        # send to device and set attributes
        self.max_r = max_r.clone().to(self.device)
        self.max_r_init = max_r.clone().to(self.device)

    def agg2latent(self, agg_state):
        return agg_state[0], None

    def _decode_step_input(self, latent_obs, agg_state_old):
        assert len(latent_obs) == 1
        r = latent_obs[0]
        max_r = agg_state_old[0]
        return r, max_r

    def step(self, latent_obs, agg_state_old):
        # decode input
        r, max_r = self._decode_step_input(
            latent_obs=latent_obs, agg_state_old=agg_state_old
        )

        # check input
        assert r.ndim == 3
        assert max_r.ndim == 2
        assert r.shape[0] == max_r.shape[0]
        assert r.shape[2] == max_r.shape[1]
        assert r.shape[1] > 0

        max_r_latent_obs, _ = torch.max(r, dim=1)  # the max of the current latent_obs
        max_r_new, _ = torch.max(torch.stack((max_r, max_r_latent_obs), axis=2), axis=2)

        return (max_r_new,)

    def update(self, latent_obs, incremental=False):
        r = latent_obs[0]
        if r.shape[1] == 0:
            return  # nothing to do

        # perform update
        if incremental:  # incremental update is not applicable with self-attention!
            agg_state_old = self.last_agg_state
        else:
            agg_state_old = self.initial_agg_state
        new_agg_state = self.step(latent_obs=latent_obs, agg_state_old=agg_state_old)

        # append new state
        self._append_agg_state(new_agg_state)


class MaxAggregatorRtoZ(MaxAggregator):
    def __init__(self, **kwargs):
        # TODO: implement two-headed r-to-z networks
        super().__init__(**kwargs)
        self.d_z = kwargs["d_z"]
        self.mean_z_mlp = MLP(
            name="agg2mean_z",
            d_in=kwargs["d_r"],
            d_out=kwargs["d_z"],
            mlp_layers=kwargs["mlp_layers_r_to_mu_z"],
            f_act=kwargs["f_act"],
            f_out=(None, {}),
            seed=kwargs["seed"],
        )
        self.cov_z_mlp = MLP(
            name="agg2cov_z",
            d_in=kwargs["d_r"],
            d_out=kwargs["d_z"],
            mlp_layers=kwargs["mlp_layers_r_to_cov_z"],
            f_act=kwargs["f_act"],
            f_out=(None, {}),
            seed=kwargs["seed"],
        )

    def agg2latent(self, agg_state):
        return (
            self.mean_z_mlp(agg_state[0]),
            output_trafo(self.cov_z_mlp(agg_state[0]), lower_bound=1e-4),
        )

    def to(self, device):
        self.device = device
        self.mean_z_mlp.to(device)
        self.cov_z_mlp.to(device)

    @property
    def parameters(self):
        return list(self.mean_z_mlp.parameters()) + list(self.cov_z_mlp.parameters())

    def save_weights(self, logpath, epoch):
        self.mean_z_mlp.save_weights(path=logpath, epoch=epoch)
        self.cov_z_mlp.save_weights(path=logpath, epoch=epoch)

    def load_weights(self, logpath, epoch):
        self.mean_z_mlp.load_weights(path=logpath, epoch=epoch)
        self.cov_z_mlp.load_weights(path=logpath, epoch=epoch)

    def delete_all_weight_files(self, logpath):
        self.mean_z_mlp.delete_all_weight_files(path=logpath)
        self.cov_z_mlp.delete_all_weight_files(path=logpath)

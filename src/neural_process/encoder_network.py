import torch

from neural_process.attention import Attention
from np_util.mlp import MLP
from np_util.output_trafo import output_trafo

# TODO: encoder networks shall inherit from a proper torch module


class EncoderNetworkBA:
    def __init__(self, **kwargs):
        self.d_x = kwargs["d_x"]
        self.d_y = kwargs["d_y"]
        self.d_r = kwargs["d_r"]
        self.arch = kwargs["arch"]
        self.f_act = kwargs["f_act"]
        self.seed = kwargs["seed"]

        # process network shapes
        self.mlp_layers_r = kwargs["mlp_layers_r"]
        self.mlp_layers_cov_r = kwargs["mlp_layers_cov_r"]

        self.r_net = self.cov_r_net = self.r_cov_r_net = None
        self.create_networks()

    def to(self, device):
        if self.arch == "separate_networks":
            self.r_net = self.r_net.to(device)
            self.cov_r_net = self.cov_r_net.to(device)
        elif self.arch == "two_heads":
            self.r_cov_r_net = self.r_cov_r_net.to(device)

    def save_weights(self, logpath, epoch):
        if self.arch == "separate_networks":
            self.r_net.save_weights(path=logpath, epoch=epoch)
            self.cov_r_net.save_weights(path=logpath, epoch=epoch)
        elif self.arch == "two_heads":
            self.r_cov_r_net.save_weights(path=logpath, epoch=epoch)

    def load_weights(self, logpath, epoch):
        if self.arch == "separate_networks":
            self.r_net.load_weights(path=logpath, epoch=epoch)
            self.cov_r_net.load_weights(path=logpath, epoch=epoch)
        elif self.arch == "two_heads":
            self.r_cov_r_net.load_weights(path=logpath, epoch=epoch)

    def create_networks(self):
        if self.arch == "separate_networks":
            self.r_net = MLP(
                name="encoder_mu",
                d_in=self.d_x + self.d_y,
                d_out=self.d_r,
                mlp_layers=self.mlp_layers_r,
                f_act=self.f_act,
                seed=self.seed,
            )
            self.cov_r_net = MLP(
                name="encoder_cov",
                d_in=self.d_x + self.d_y,
                d_out=self.d_r,
                mlp_layers=self.mlp_layers_cov_r,
                f_act=self.f_act,
                seed=self.seed,
            )
        elif self.arch == "two_heads":
            self.r_cov_r_net = MLP(
                name="encoder_mu_cov",
                d_in=self.d_x + self.d_y,
                d_out=2 * self.d_r,
                mlp_layers=self.mlp_layers_r,  # ignore self.mlp_layers_cov_r
                f_act=self.f_act,
                seed=self.seed,
            )
        else:
            raise ValueError("Unknown encoder network type!")

    def encode(self, x, y):
        assert x.ndim == y.ndim == 3

        # prepare input to encoder network
        encoder_input = torch.cat((x, y), dim=2)

        # encode
        if self.arch == "separate_networks":
            r, cov_r = self.r_net(encoder_input), self.cov_r_net(encoder_input)
        elif self.arch == "two_heads":
            mu_r_cov_r = self.r_cov_r_net(encoder_input)
            r = mu_r_cov_r[:, :, : self.d_r]
            cov_r = mu_r_cov_r[:, :, self.d_r :]

        cov_r = output_trafo(cov_r, lower_bound=1e-4)

        return r, cov_r

    @property
    def parameters(self):
        if self.arch == "separate_networks":
            return list(self.r_net.parameters()) + list(self.cov_r_net.parameters())
        elif self.arch == "two_heads":
            return list(self.r_cov_r_net.parameters())


class EncoderNetworkMA:
    def __init__(self, **kwargs):
        self.d_x = kwargs["d_x"]
        self.d_y = kwargs["d_y"]
        self.d_r = kwargs["d_r"]
        self.mlp_layers = kwargs["mlp_layers_r"]
        self.f_act = kwargs["f_act"]
        self.seed = kwargs["seed"]

        self.r_net = None
        self.create_networks()

    def to(self, device):
        self.r_net = self.r_net.to(device)

    def save_weights(self, logpath, epoch):
        self.r_net.save_weights(path=logpath, epoch=epoch)

    def load_weights(self, logpath, epoch):
        self.r_net.load_weights(path=logpath, epoch=epoch)

    def create_networks(self):
        self.r_net = MLP(
            name="encoder_r",
            d_in=self.d_x + self.d_y,
            d_out=self.d_r,
            mlp_layers=self.mlp_layers,
            f_act=self.f_act,
            seed=self.seed,
        )

    def encode(self, x, y):
        assert x.ndim == y.ndim == 3

        # prepare input to encoder network
        encoder_input = torch.cat((x, y), dim=2)

        # encode
        r = self.r_net(encoder_input)

        return (r,)

    @property
    def parameters(self):
        return list(self.r_net.parameters())


class EncoderNetworkSAMA(EncoderNetworkMA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attention = Attention(**kwargs)

    def to(self, device):
        super().to(device=device)
        self.attention.set_device(device=device)

    def save_weights(self, logpath, epoch):
        super().save_weights(logpath=logpath, epoch=epoch)
        self.attention.save_weights(logpath=logpath, epoch=epoch)

    def load_weights(self, logpath, epoch):
        super().load_weights(logpath=logpath, epoch=epoch)
        self.attention.load_weights(logpath=logpath, epoch=epoch)

    def encode(self, x, y):
        r = super().encode(x=x, y=y)[0]
        if x.shape[1] > 0:  # only apply self-attention if context set is not empty
            r = self.attention(q=x, k=x, v=r)

        return (r,)

    @property
    def parameters(self):
        return super().parameters + self.attention.parameters

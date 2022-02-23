# ******************************************************************
# mlp.py
# A simple multi-layer perceptron in pytorch.
# ******************************************************************

import math
import os

import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.modules.container import ModuleList


def tanh2(x, min_y, max_y):
    scale_x = 1 / ((max_y - min_y) / 2)
    return (max_y - min_y) / 2 * (torch.tanh(x * scale_x) + 1.0) + min_y


class MLP(nn.Module):
    def __init__(
        self,
        name,
        d_in=None,
        d_out=None,
        mlp_layers=None,
        shape=None,
        d_avg=None,
        n_hidden=None,
        f_act=None,
        f_out=(None, {}),
        seed=None,
        use_standard_initialization=True,
        std_weight=None,
        std_bias=None,
    ):
        super(MLP, self).__init__()

        self.name = name

        self.function_name_mappings = {
            "tanh": torch.tanh,
            "tanh2": tanh2,
            "relu": torch.relu,
            "softplus": F.softplus,
            "exp": torch.exp,
            "None": None,
        }

        self.d_in = d_in
        self.d_out = d_out
        if mlp_layers is not None:
            # either provide the arch spec directly via mlp_layers
            assert shape is None
            assert d_avg is None
            assert n_hidden is None
            self.arch_spec = mlp_layers
        else:
            # or compute the arch spec from shape, d_avg, and n_hidden
            assert mlp_layers is None
            self.arch_spec = self.compute_arch_spec(
                shape=shape, d_avg=d_avg, n_hidden=n_hidden
            )
        self.n_hidden_layers = len(self.arch_spec)
        self.is_linear = self.n_hidden_layers == 0
        self.f_act = self.function_name_mappings[f_act] if f_act is not None else None
        self.f_out = (
            self.function_name_mappings[f_out[0]] if f_out[0] is not None else None
        )
        self.f_out_params = f_out[1]
        self.seed = seed

        self.layers = None
        self.create_network()
        if not use_standard_initialization:
            self.initialize_weights(std_weight=std_weight, std_bias=std_bias)

    def create_network(self):
        # process architecture
        if not self.is_linear:
            assert self.f_act is not None

        # seeding
        if self.seed is not None:
            torch.manual_seed(self.seed)

        # define the network
        if self.is_linear:
            self.layers = ModuleList(
                [nn.Linear(in_features=self.d_in, out_features=self.d_out)]
            )
        else:
            self.layers = ModuleList(
                [nn.Linear(in_features=self.d_in, out_features=self.arch_spec[0])]
            )
            for i in range(1, len(self.arch_spec)):
                self.layers.append(
                    nn.Linear(
                        in_features=self.layers[-1].out_features,
                        out_features=self.arch_spec[i],
                    )
                )
            self.layers.append(
                nn.Linear(
                    in_features=self.layers[-1].out_features, out_features=self.d_out
                )
            )

    def forward(self, x, output_layer=None):
        if output_layer is None:
            output_layer = self.n_hidden_layers + 1
        assert 0 <= output_layer <= len(self.arch_spec) + 1

        y = x
        if output_layer == 0:
            return y

        if self.is_linear:
            y = (
                self.layers[0](y)
                if self.f_out is None
                else self.f_out(self.layers[0](y), **self.f_out_params)
            )
        else:
            # do not iterate directly over self.layers, this is slow using ModuleList
            for i in range(self.n_hidden_layers):
                y = self.f_act(self.layers[i](y))
                if i + 1 == output_layer:
                    return y
            y = (
                self.layers[-1](y)
                if self.f_out is None
                else self.f_out(self.layers[-1](y), **self.f_out_params)
            )

        return y

    def save_weights(self, path, epoch):
        with open(
            os.path.join(path, self.name + "_weights_{:d}".format(epoch)), "wb"
        ) as f:
            torch.save(self.state_dict(), f)

    def load_weights(self, path, epoch):
        self.load_state_dict(
            torch.load(os.path.join(path, self.name + "_weights_{:d}".format(epoch)))
        )

    def delete_all_weight_files(self, path):
        for file in os.listdir(path):
            if file.startswith(self.name + "_weights"):
                os.remove(os.path.join(path, file))

    @staticmethod
    def compute_arch_spec(shape, d_avg, n_hidden):
        assert d_avg >= 0
        assert -1.0 <= shape <= 1.0
        assert n_hidden >= 1
        shape = shape * d_avg  # we want the user to provide shape \in [-1, +1]

        arch_spec = []
        for i in range(n_hidden):
            # compute real-valued 'position' x of current layer (x \in (-1, 1))
            x = 2 * i / (n_hidden - 1) - 1 if n_hidden != 1 else 0.0

            # compute number of units in current layer
            d = shape * x + d_avg
            d = int(math.floor(d))
            if d == 0:  # occurs if shape == -d_avg or shape == d_avg
                d = 1
            arch_spec.append(d)

        return arch_spec


class LearnableTensor(nn.Module):
    """
    Learns a single tensor, with convenience functionality to save, load, add batch dims.
    """

    def __init__(self, name, init_mean, uniform_init_delta=None, seed=None):
        super(LearnableTensor, self).__init__()

        self.name = name
        value = torch.tensor(init_mean)
        if uniform_init_delta is not None:
            if seed is not None:
                torch.manual_seed(seed)

            eps = torch.empty(value.shape)
            eps.uniform_(0, 1.0)
            eps = (eps - 0.5) * 2 * uniform_init_delta
            value += eps
        self.value = nn.Parameter(value)

    def forward(self, batch_shape=None):
        """Appends batch_shape-like dimensions to front, if provided."""
        if batch_shape is not None:
            return self.value.expand(batch_shape + self.value.shape)
        else:
            return self.value

    def save_weights(self, path, epoch):
        with open(
            os.path.join(path, self.name + "_weights_{:d}".format(epoch)), "wb"
        ) as f:
            torch.save(self.state_dict(), f)

    def load_weights(self, path, epoch):
        map_location = "cpu" if not torch.cuda.is_available() else None
        self.load_state_dict(
            torch.load(
                os.path.join(path, self.name + "_weights_{:d}".format(epoch)),
                map_location=map_location,
            )
        )

    def delete_all_weight_files(self, path):
        for file in os.listdir(path):
            if file.startswith(self.name + "_weights"):
                os.remove(os.path.join(path, file))

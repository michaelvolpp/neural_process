# source: https://github.com/deepmind/neural-processes

import os
import torch

from np_util.mlp import MLP


class UniformAttention:
    def __init__(self, **kwargs):
        pass

    def __call__(self, q, k, v):
        # Computes the mean of the values for each query, independent of the key.
        # If used as self-attention and combined with mean aggregation, this is
        # equivalent to no self-attention (take mean twice).
        # q = [B, m, d_k]
        # v = [B, n, d_v]

        assert q.ndim == v.ndim == 3
        assert q.shape[0] == k.shape[0] == v.shape[0]
        assert k.shape[1] == v.shape[1]
        assert q.shape[2] == k.shape[2]
        if v.shape[1] == 0:
            raise NotImplementedError

        rep = torch.mean(v, dim=1, keepdim=True)  # [B, 1, d_v]
        rep = rep.expand(q.shape[0], q.shape[1], v.shape[-1])  # [B, m, d_v]

        return rep

    def set_device(self, device):
        pass

    def save_weights(self, logpath, epoch):
        pass

    def load_weights(self, logpath, epoch):
        pass

    @property
    def parameters(self):
        return []


class LaplaceAttention:
    def __init__(self, **kwargs):
        pass

    def __call__(self, q, k, v):
        # q = [B, m, d_k]
        # k = [B, n, d_k]
        # v = [B, n, d_v]

        assert q.ndim == v.ndim == 3
        assert q.shape[0] == k.shape[0] == v.shape[0]
        assert k.shape[1] == v.shape[1]
        assert q.shape[2] == k.shape[2]
        if v.shape[1] == 0:
            raise NotImplementedError

        k = k[:, None, :, :]  # [B, 1, n, d_k]
        q = q[:, :, None, :]  # [B, m, 1, d_k]
        w = -torch.abs((k - q))  # [B, m, n, d_k]
        w = torch.sum(w, dim=-1)  # [B, m, n]
        w = torch.nn.functional.softmax(w, dim=-1)  # [B, m, n]
        rep = torch.einsum("bik,bkj->bij", w, v)  # [B, m, d_v]

        return rep

    def set_device(self, device):
        pass

    def save_weights(self, logpath, epoch):
        pass

    def load_weights(self, logpath, epoch):
        pass

    @property
    def parameters(self):
        return []


class DotProductAttention:
    def __init__(self, **kwargs):
        pass

    def __call__(self, q, k, v):
        # q = [B, m, d_k]
        # k = [B, n, d_k]
        # v = [B, n, d_v]

        assert q.ndim == v.ndim == 3
        assert q.shape[0] == k.shape[0] == v.shape[0]
        assert k.shape[1] == v.shape[1]
        assert q.shape[2] == k.shape[2]
        if v.shape[1] == 0:
            raise NotImplementedError

        d_k = torch.tensor(q.shape[2], dtype=torch.float)
        w = torch.einsum("bik,bjk->bij", q, k) / torch.sqrt(d_k)  # [B, m, n]
        w = torch.nn.functional.softmax(w, dim=-1)  # [B, m, n]
        rep = torch.einsum("bik,bkj->bij", w, v)  # [B, m, d_v]

        return rep

    def set_device(self, device):
        pass

    def save_weights(self, logpath, epoch):
        pass

    def load_weights(self, logpath, epoch):
        pass

    @property
    def parameters(self):
        return []


class MultiheadAttention:
    def __init__(self, d_k, d_v, seed, f_act, num_heads=8, **kwargs):
        self.d_k = d_k
        self.d_v = d_v
        self.f_act = f_act
        self.seed = seed
        assert d_v % num_heads == 0, "d_v has to be divisible by num_heads!"

        self.q_k_embed_net = None
        self.create_networks()

        self.multihead_attention_torch = torch.nn.MultiheadAttention(
            embed_dim=self.d_v, num_heads=num_heads
        )
        self.multihead_attention_name = "multihead_attention"

    def __call__(self, q, k, v):
        # q = [B, m, d_k]
        # k = [B, n, d_k]
        # v = [B, n, d_v]

        assert q.ndim == v.ndim == 3
        assert q.shape[0] == k.shape[0] == v.shape[0]
        assert k.shape[1] == v.shape[1]
        assert q.shape[2] == k.shape[2]
        if v.shape[1] == 0:
            raise NotImplementedError

        # map to embedding space
        q, k = self.q_k_embed_net(q), self.q_k_embed_net(k)

        # apply multihead attention
        rep, _ = self.multihead_attention_torch(
            query=q.permute(1, 0, 2),  # the second dim is the batch dim here
            key=k.permute(1, 0, 2),
            value=v.permute(1, 0, 2),
            need_weights=False,
        )
        rep = rep.permute(1, 0, 2)  # make batch dim the first dim again

        return rep

    def create_networks(self):
        self.q_k_embed_net = MLP(
            name="q_k_embed_net_multihead_attention",
            d_in=self.d_k,
            d_out=self.d_v,
            mlp_layers=[(self.d_k + self.d_v) // 2],
            f_act=self.f_act,
            seed=self.seed,
        )

    def set_device(self, device):
        self.multihead_attention_torch.to(device)
        self.q_k_embed_net.to(device)

    def save_weights(self, logpath, epoch):
        self.q_k_embed_net.save_weights(path=logpath, epoch=epoch)
        if epoch is not None:
            with open(
                os.path.join(
                    logpath,
                    self.multihead_attention_name + "_weights_{:d}".format(epoch),
                ),
                "wb",
            ) as f:
                torch.save(self.multihead_attention_torch.state_dict(), f)
        else:
            with open(
                os.path.join(logpath, self.multihead_attention_name + "_weights"), "wb"
            ) as f:
                torch.save(self.multihead_attention_torch.state_dict(), f)

    def load_weights(self, logpath, epoch):
        self.q_k_embed_net.load_weights(path=logpath, epoch=epoch)
        if epoch is not None:
            super(
                torch.nn.MultiheadAttention, self.multihead_attention_torch
            ).load_state_dict(
                torch.load(
                    os.path.join(
                        logpath,
                        self.multihead_attention_name + "_weights_{:d}".format(epoch),
                    )
                )
            )
        else:
            super(
                torch.nn.MultiheadAttention, self.multihead_attention_torch
            ).load_state_dict(
                torch.load(
                    os.path.join(logpath, self.multihead_attention_name + "_weights")
                )
            )

    @property
    def parameters(self):
        params = list(self.multihead_attention_torch.parameters())
        params += list(self.q_k_embed_net.parameters())
        return params


f_attention_dict = {
    "uniform": UniformAttention,
    "laplace": LaplaceAttention,
    "dot_product": DotProductAttention,
    "multihead": MultiheadAttention,
}


class Attention:
    def __init__(self, **kwargs):
        self.f_attention = f_attention_dict[kwargs["attention_type"]](
            d_k=kwargs["d_x"], d_v=kwargs["d_r"], **kwargs
        )

    def __call__(self, q, k, v):
        return self.f_attention(q=q, k=k, v=v)

    def set_device(self, device):
        self.f_attention.set_device(device=device)

    def save_weights(self, logpath, epoch):
        self.f_attention.save_weights(logpath=logpath, epoch=epoch)

    def load_weights(self, logpath, epoch):
        self.f_attention.load_weights(logpath=logpath, epoch=epoch)

    @property
    def parameters(self):
        return self.f_attention.parameters

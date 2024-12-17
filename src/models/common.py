import math

import torch.nn as nn
from torch import einsum
from einops import rearrange


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=(kernel_size // 2),
        bias=bias,
    )


class ResBlock(nn.Module):
    def __init__(
        self,
        conv,
        n_feats,
        kernel_size,
        bias=True,
        batch_norm=False,
        act=nn.ReLU(True),
        res_scale=1,
    ):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale

        modules = []
        for i in range(2):
            modules.append(
                conv(
                    in_channels=n_feats,
                    out_channels=n_feats,
                    kernel_size=kernel_size,
                    bias=bias,
                )
            )
            if i == 0:  # Add an activation function after the first convolution
                modules.append(act)
            if batch_norm:
                modules.append(nn.BatchNorm1d(n_feats))

        self.body = nn.Sequential(*modules)

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


# Taken from: https://github.com/serkansulun/pytorch-pixelshuffle1d/blob/master/pixelshuffle1d.py
class PixelShuffle1D(nn.Module):
    """
    1D pixel shuffler. https://arxiv.org/pdf/1609.05158.pdf
    Upscales sample length, downscales channel length
    "short" is input, "long" is output
    """

    def __init__(self, upscale_factor):
        super(PixelShuffle1D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        short_channel_len = x.shape[1]
        short_width = x.shape[2]

        long_channel_len = short_channel_len // self.upscale_factor
        long_width = self.upscale_factor * short_width

        x = x.contiguous().view(
            [batch_size, self.upscale_factor, long_channel_len, short_width]
        )
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, long_channel_len, long_width)

        return x


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 2 * n_feats, 3, bias))
                m.append(PixelShuffle1D(2))
                if bn:
                    m.append(nn.BatchNorm1d(n_feats))
                if act == "relu":
                    m.append(nn.ReLU(True))
                elif act == "prelu":
                    m.append(nn.PReLU(n_feats))

        else:
            m.append(
                conv(
                    in_channels=n_feats,
                    out_channels=scale * n_feats,
                    kernel_size=scale,
                    bias=bias,
                )
            )
            m.append(PixelShuffle1D(scale))
            if bn:
                m.append(nn.BatchNorm1d(n_feats))
            if act == "relu":
                m.append(nn.ReLU(True))
            elif act == "prelu":
                m.append(nn.PReLU(n_feats))

        super(Upsampler, self).__init__(*m)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm2(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, x2, **kwargs):
        return self.fn(self.norm(x), self.norm2(x2), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = self.attend(dots)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x_q, x_kv):
        _, _, dim, heads = *x_q.shape, self.heads
        _, _, dim_large = x_kv.shape

        assert dim == dim_large

        q = self.to_q(x_q)

        q = rearrange(q, "b n (h d) -> b h n d", h=heads)

        kv = self.to_kv(x_kv).chunk(2, dim=-1)

        k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=heads), kv)

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = self.attend(dots)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out)

from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
)


class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb=None):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        
        self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """

        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        
        # implement the attention layer here
        d = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, width // self.n_heads, length).split(d, dim=1)

        # attention weight w = softmax(q^Tk / sqrt(d))
        # better performance compared to w = th.softmax(th.einsum("ikj,ikl->ijl", q, k) / math.sqrt(d), dim=-1)
        denom_sqrt = 1 / math.sqrt(math.sqrt(d)) 
        w = th.softmax(th.einsum("ikj,ikl->ijl", q * denom_sqrt, k * denom_sqrt), dim=-1)

        # output = vw^T, reshape to match the return size
        return th.einsum("ijl,ikl->ijk", v, w).reshape(bs, width // 3, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)

class UNet(nn.Module):

    def __init__(self, channels, num_classes=None, dropout=0.0):
        super(UNet, self).__init__()

        self.emb_channels = 512
        self.time_embed = nn.Sequential(
            linear(128, 512),
            nn.SiLU(),
            linear(512, 512),
        )
        
        # class condition
        self.num_classes = num_classes
        if num_classes is not None:
            # implement the class embedding here
            self.label_emb = nn.Embedding(num_embeddings=num_classes, embedding_dim=self.emb_channels)

        self.input_conv = nn.Conv2d(3, channels, 3, padding=1)

        self.block1 = ResBlock(channels, self.emb_channels, dropout, channels, use_conv=False, use_scale_shift_norm=True)
        self.block2 = ResBlock(channels, self.emb_channels, dropout, channels, use_conv=False, use_scale_shift_norm=True)
        self.block3 = ResBlock(channels, self.emb_channels, dropout, channels, use_conv=False, use_scale_shift_norm=True)

        self.block4 = ResBlock(channels, self.emb_channels, dropout, channels, use_conv=False, use_scale_shift_norm=True, down=True)
        self.block5 = ResBlock(channels, self.emb_channels, dropout, channels, use_conv=False, use_scale_shift_norm=True)
        self.block6 = ResBlock(channels, self.emb_channels, dropout, channels, use_conv=False, use_scale_shift_norm=True)
        self.block7 = ResBlock(channels, self.emb_channels, dropout, channels, use_conv=False, use_scale_shift_norm=True)
        
        self.block8 = ResBlock(channels, self.emb_channels, dropout, 2*channels, use_conv=False, use_scale_shift_norm=True, down=True)
        self.block9 = ResBlock(2*channels, self.emb_channels, dropout, 2*channels, use_conv=False, use_scale_shift_norm=True)
        self.block10 = ResBlock(2*channels, self.emb_channels, dropout, 2*channels, use_conv=False, use_scale_shift_norm=True)
        self.block11 = ResBlock(2*channels, self.emb_channels, dropout, 2*channels, use_conv=False, use_scale_shift_norm=True)

        self.block12 = ResBlock(2*channels, self.emb_channels, dropout, 3*channels, use_conv=False, use_scale_shift_norm=True, down=True)
        # add attention layers from here
        self.block13 = ResBlock(3*channels, self.emb_channels, dropout, 3*channels, use_conv=False, use_scale_shift_norm=True)
        self.attn_block13 = AttentionBlock(3*channels, num_heads=4)
        self.block14 = ResBlock(3*channels, self.emb_channels, dropout, 3*channels, use_conv=False, use_scale_shift_norm=True)
        self.attn_block14 = AttentionBlock(3*channels, num_heads=4)
        self.block15 = ResBlock(3*channels, self.emb_channels, dropout, 3*channels, use_conv=False, use_scale_shift_norm=True)
        self.attn_block15 = AttentionBlock(3*channels, num_heads=4)

        self.block16 = ResBlock(3*channels, self.emb_channels, dropout, 4*channels, use_conv=False, use_scale_shift_norm=True, down=True)
        self.attn_block16 = AttentionBlock(4*channels, num_heads=4)
        self.block17 = ResBlock(4*channels, self.emb_channels, dropout, 4*channels, use_conv=False, use_scale_shift_norm=True)
        self.attn_block17 = AttentionBlock(4*channels, num_heads=4)
        self.block18 = ResBlock(4*channels, self.emb_channels, dropout, 4*channels, use_conv=False, use_scale_shift_norm=True)
        self.attn_block18 = AttentionBlock(4*channels, num_heads=4)
        self.block19 = ResBlock(4*channels, self.emb_channels, dropout, 4*channels, use_conv=False, use_scale_shift_norm=True)
        self.attn_block19 = AttentionBlock(4*channels, num_heads=4)

        # middle blocks
        self.mid_block1 = ResBlock(4*channels, self.emb_channels, dropout, 4*channels, use_conv=False, use_scale_shift_norm=True)
        self.mid_attn_block1 = AttentionBlock(4*channels, num_heads=4)
        self.mid_block2 = ResBlock(4*channels, self.emb_channels, dropout, 4*channels, use_conv=False, use_scale_shift_norm=True)

        # decoder
        self.dec_block19 = ResBlock(8*channels, self.emb_channels, dropout, 4*channels, use_conv=False, use_scale_shift_norm=True)
        self.dec_attn_block19 = AttentionBlock(4*channels, num_heads=4)
        self.dec_block18 = ResBlock(8*channels, self.emb_channels, dropout, 4*channels, use_conv=False, use_scale_shift_norm=True)
        self.dec_attn_block18 = AttentionBlock(4*channels, num_heads=4)
        self.dec_block17 = ResBlock(8*channels, self.emb_channels, dropout, 4*channels, use_conv=False, use_scale_shift_norm=True)
        self.dec_attn_block17 = AttentionBlock(4*channels, num_heads=4)
        self.dec_block16 = ResBlock(8*channels, self.emb_channels, dropout, 3*channels, use_conv=False, use_scale_shift_norm=True, up=True)
        self.dec_attn_block16 = AttentionBlock(3*channels, num_heads=4)

        self.dec_block15 = ResBlock(6*channels, self.emb_channels, dropout, 3*channels, use_conv=False, use_scale_shift_norm=True)
        self.dec_attn_block15 = AttentionBlock(3*channels, num_heads=4)
        self.dec_block14 = ResBlock(6*channels, self.emb_channels, dropout, 3*channels, use_conv=False, use_scale_shift_norm=True)
        self.dec_attn_block14 = AttentionBlock(3*channels, num_heads=4)
        self.dec_block13 = ResBlock(6*channels, self.emb_channels, dropout, 3*channels, use_conv=False, use_scale_shift_norm=True)
        self.dec_attn_block13 = AttentionBlock(3*channels, num_heads=4)
        self.dec_block12 = ResBlock(6*channels, self.emb_channels, dropout, 2*channels, use_conv=False, use_scale_shift_norm=True, up=True)

        self.dec_block11 = ResBlock(4*channels, self.emb_channels, dropout, 2*channels, use_conv=False, use_scale_shift_norm=True)
        self.dec_block10 = ResBlock(4*channels, self.emb_channels, dropout, 2*channels, use_conv=False, use_scale_shift_norm=True)
        self.dec_block9 = ResBlock(4*channels, self.emb_channels, dropout, 2*channels, use_conv=False, use_scale_shift_norm=True)
        self.dec_block8 = ResBlock(4*channels, self.emb_channels, dropout, channels, use_conv=False, use_scale_shift_norm=True, up=True)

        self.dec_block7 = ResBlock(2*channels, self.emb_channels, dropout, channels, use_conv=False, use_scale_shift_norm=True)
        self.dec_block6 = ResBlock(2*channels, self.emb_channels, dropout, channels, use_conv=False, use_scale_shift_norm=True)
        self.dec_block5 = ResBlock(2*channels, self.emb_channels, dropout, channels, use_conv=False, use_scale_shift_norm=True)
        self.dec_block4 = ResBlock(2*channels, self.emb_channels, dropout, channels, use_conv=False, use_scale_shift_norm=True, up=True)

        self.dec_block3 = ResBlock(2*channels, self.emb_channels, dropout, channels, use_conv=False, use_scale_shift_norm=True)
        self.dec_block2 = ResBlock(2*channels, self.emb_channels, dropout, channels, use_conv=False, use_scale_shift_norm=True)
        self.dec_block1 = ResBlock(2*channels, self.emb_channels, dropout, channels, use_conv=False, use_scale_shift_norm=True)
        self.dec_block0 = ResBlock(2*channels, self.emb_channels, dropout, channels, use_conv=False, use_scale_shift_norm=True)

        self.out = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, 6, 3, 1, 1),
        )
    
    def timestep_embedding(self, timesteps, channels):
        # implement the time embedding here
        # note that e^ln(10000) = 10000 and a^(bc) = (a^b)^c
        half_channels = channels // 2
        denom = th.exp(-math.log(10000) * th.arange(end=half_channels, dtype=th.float32) / half_channels).cuda()
        vals = th.unsqueeze(timesteps, dim=1) * th.unsqueeze(denom, dim=0)
        
        # compute cosine and sine values
        return th.cat([th.cos(vals), th.sin(vals)], dim=-1)

    def forward(self, x, timesteps, y=None):
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        t_emb = self.timestep_embedding(timesteps, 128)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            # add the class embedding to time embedding here for class conditioned image generation
            emb += self.label_emb(y)

        # encoder
        x0 = self.input_conv(x)
        x1 = self.block1(x0, emb)
        x2 = self.block2(x1, emb)
        x3 = self.block3(x2, emb)

        x4 = self.block4(x3, emb)
        x5 = self.block5(x4, emb)
        x6 = self.block6(x5, emb)
        x7 = self.block7(x6, emb)

        x8 = self.block8(x7, emb)
        x9 = self.block9(x8, emb)
        x10 = self.block10(x9, emb)
        x11 = self.block11(x10, emb)

        x12 = self.block12(x11, emb)
        x13 = self.attn_block13(self.block13(x12, emb))
        x14 = self.attn_block14(self.block14(x13, emb))
        x15 = self.attn_block15(self.block15(x14, emb))

        x16 = self.attn_block16(self.block16(x15, emb))
        x17 = self.attn_block17(self.block17(x16, emb))
        x18 = self.attn_block18(self.block18(x17, emb))
        x19 = self.attn_block19(self.block19(x18, emb))

        # middle blocks
        x_mid = self.mid_block1(x19, emb)
        x_mid = self.mid_attn_block1(x_mid)
        x_mid = self.mid_block2(x_mid, emb)

        # decoder
        dec_x19 = self.dec_attn_block19(self.dec_block19(th.cat([x_mid, x19], dim=1), emb))
        dec_x18 = self.dec_attn_block18(self.dec_block18(th.cat([dec_x19, x18], dim=1), emb))
        dec_x17 = self.dec_attn_block17(self.dec_block17(th.cat([dec_x18, x17], dim=1), emb))
        dec_x16 = self.dec_attn_block16(self.dec_block16(th.cat([dec_x17, x16], dim=1), emb))

        dec_x15 = self.dec_attn_block15(self.dec_block15(th.cat([dec_x16, x15], dim=1), emb))
        dec_x14 = self.dec_attn_block14(self.dec_block14(th.cat([dec_x15, x14], dim=1), emb))
        dec_x13 = self.dec_attn_block13(self.dec_block13(th.cat([dec_x14, x13], dim=1), emb))
        dec_x12 = self.dec_block12(th.cat([dec_x13, x12], dim=1), emb)

        dec_x11 = self.dec_block11(th.cat([dec_x12, x11], dim=1), emb)
        dec_x10 = self.dec_block10(th.cat([dec_x11, x10], dim=1), emb)
        dec_x9 = self.dec_block9(th.cat([dec_x10, x9], dim=1), emb)
        dec_x8 = self.dec_block8(th.cat([dec_x9, x8], dim=1), emb)

        dec_x7 = self.dec_block7(th.cat([dec_x8, x7], dim=1), emb)
        dec_x6 = self.dec_block6(th.cat([dec_x7, x6], dim=1), emb)
        dec_x5 = self.dec_block5(th.cat([dec_x6, x5], dim=1), emb)
        dec_x4 = self.dec_block4(th.cat([dec_x5, x4], dim=1), emb)

        dec_x3 = self.dec_block3(th.cat([dec_x4, x3], dim=1), emb)
        dec_x2 = self.dec_block2(th.cat([dec_x3, x2], dim=1), emb)
        dec_x1 = self.dec_block1(th.cat([dec_x2, x1], dim=1), emb)
        dec_x0 = self.dec_block0(th.cat([dec_x1, x0], dim=1), emb)

        return self.out(dec_x0)

def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])
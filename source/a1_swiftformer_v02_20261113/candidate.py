"""A1-swiftformer-v0.2: repeated two-stage D-WPO -> DualStage blocks.

The original A1 E2A bottleneck, E3 skip, SwiftFormer-S encoder and head are
retained. Three coupled blocks span D3->D2, D2->D1 and D1->D0. Each block
applies a scale-specific D-WPO, then upsamples, fuses the next encoder skip,
and applies a channel-configured DualStage.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from operations.encoder_swap_A016_20261107.encoder_swap_candidate import (
    build_candidate as build_encoder_swap,
)
from source.decoder_dual_a_20261106 import candidate as dual_source
from source.three_module_impl_20261104.candidate import DirectionalWaveOperator


class WidthMambaCorrection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        from mamba_ssm.modules.mamba_simple import Mamba

        self.norm = nn.LayerNorm(channels)
        self.mixer = Mamba(d_model=channels, d_state=16, d_conv=4, expand=2)

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        return self.mixer(self.norm(tokens)).transpose(1, 2).reshape(b, c, h, w)


class ScaledDualStage(nn.Module):
    """A1 D2 Mamba + RepDW7 correction, generalized to stage widths."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        joint = dual_source.joint
        self.project = nn.Conv2d(in_channels, out_channels, 1, bias=True)
        self.mamba = WidthMambaCorrection(out_channels)
        self.rep = nn.Sequential(
            joint.LN(out_channels), joint.Rep(out_channels, 7, False),
            nn.GELU(), joint.pw(out_channels, out_channels),
        )
        self.raw_m = nn.Parameter(torch.tensor(math.atanh(0.1)))
        self.raw_r = nn.Parameter(torch.tensor(math.atanh(0.1)))

    def forward(self, x):
        u = self.project(x)
        return u + 0.1 * self.raw_m.tanh() * self.mamba(u) + 0.1 * self.raw_r.tanh() * self.rep(u)


class WaveDualBlock(nn.Module):
    """D-WPO at one scale followed by upsample/skip fusion and DualStage."""

    def __init__(self, channels, resolution, upsample, out_channels, with_e3=False):
        super().__init__()
        self.wave = DirectionalWaveOperator(
            res=resolution, dim=channels, hidden_dim=channels, num_heads=1,
        )
        self.alpha_embed = nn.Parameter(torch.full(
            (resolution, resolution // 2 + 1, 1), math.log(math.expm1(0.05)),
        ))
        self.wave_scale = nn.Parameter(torch.full((1, channels, 1, 1), 1e-3))
        self.upsample = upsample
        self.with_e3 = with_e3
        self.dual = ScaledDualStage(2 * out_channels, out_channels)

    def forward(self, x, skip, e3=None):
        x = x + self.wave_scale * self.wave(x, self.alpha_embed)
        x = self.upsample(x)
        if self.with_e3:
            if e3 is None:
                raise ValueError("D1->D0 block requires the retained A1 E3 skip")
            skip, x = e3(skip, x)
        return self.dual(torch.cat((x, skip), dim=1))


class CoupledDecoder(nn.Module):
    def __init__(self, old):
        super().__init__()
        self.e2 = old.e2  # Original A1 E2A; no bottleneck modification in v0.2.
        self.e3 = old.e3
        self.head = old.head
        self.blocks = nn.ModuleList([
            WaveDualBlock(512, 8, old.ups[2], 256),
            WaveDualBlock(256, 16, old.ups[1], 128),
            WaveDualBlock(128, 32, old.ups[0], 64, with_e3=True),
        ])

    def forward(self, features):
        x = self.e2(features[3])
        x = self.blocks[0](x, features[2])
        x = self.blocks[1](x, features[1])
        x = self.blocks[2](x, features[0], self.e3)
        return torch.sigmoid(F.interpolate(
            self.head(x), size=(256, 256), mode="bilinear", align_corners=False,
        ))


def build_candidate(pretrained=True, encoder="swiftformer_s"):
    if encoder != "swiftformer_s":
        raise ValueError(f"only swiftformer_s is defined, got {encoder!r}")
    model = build_encoder_swap("swiftformer_s__A1", pretrained=pretrained)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261114)
        model.decoder = CoupledDecoder(model.decoder)
    return model

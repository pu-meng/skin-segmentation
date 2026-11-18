"""A1-swiftformer-v0.1 exploratory variant: E2A->D-WPO deep block plus DualStage at D3-D0.

This is a separate candidate. It does not alter the existing A1 builders.
The encoder is SwiftFormer-S; the A1 E3 skip and segmentation head are retained.
"""
import math

import torch
from torch import nn

from operations.encoder_swap_A016_20261107.encoder_swap_candidate import (
    build_candidate as build_encoder_swap,
)
from source.decoder_dual_a_20261106 import candidate as dual_source
from source.three_module_impl_20261104.candidate import DStage


class WidthMambaCorrection(nn.Module):
    """Mamba correction on row-major flattened spatial tokens."""

    def __init__(self, channels):
        super().__init__()
        from mamba_ssm.modules.mamba_simple import Mamba

        self.norm = nn.LayerNorm(channels)
        self.mixer = Mamba(d_model=channels, d_state=16, d_conv=4, expand=2)

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        correction = self.mixer(self.norm(tokens))
        return correction.transpose(1, 2).reshape(b, c, h, w)


class ScaledDualStage(nn.Module):
    """Channel-configurable Mamba + RepDW7 residual stage."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        joint = dual_source.joint
        self.project = nn.Conv2d(in_channels, out_channels, 1, bias=True)
        self.mamba = WidthMambaCorrection(out_channels)
        self.rep = nn.Sequential(
            joint.LN(out_channels),
            joint.Rep(out_channels, 7, False),
            nn.GELU(),
            joint.pw(out_channels, out_channels),
        )
        # tanh(raw)=0.1 initially; multiplying by 0.1 gives a 0.01 initial scale.
        self.raw_m = nn.Parameter(torch.tensor(math.atanh(0.1)))
        self.raw_r = nn.Parameter(torch.tensor(math.atanh(0.1)))

    def forward(self, x):
        u = self.project(x)
        return (
            u
            + 0.1 * self.raw_m.tanh() * self.mamba(u)
            + 0.1 * self.raw_r.tanh() * self.rep(u)
        )


class DeepFeatureProcessingBlock(nn.Module):
    """Code-level composition E2A followed by the existing D-WPO DStage."""

    def __init__(self, e2a):
        super().__init__()
        self.bottleneck = e2a
        self.dwpo = DStage()

    def forward(self, x):
        return self.dwpo(self.bottleneck(x))


def build_candidate(pretrained=True, encoder="swiftformer_s"):
    """Build SwiftFormer-S + A1 E3/head with the proposed decoder changes."""
    if encoder != "swiftformer_s":
        raise ValueError(f"only swiftformer_s is defined, got {encoder!r}")

    # This builder preserves the exact A1 decoder/skip/head and installs SwiftFormer-S.
    model = build_encoder_swap("swiftformer_s__A1", pretrained=pretrained)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261113)
        model.decoder.e2 = DeepFeatureProcessingBlock(model.decoder.e2)
        model.decoder.stages[3] = ScaledDualStage(512, 512)
        model.decoder.stages[2] = ScaledDualStage(512, 256)
        model.decoder.stages[1] = ScaledDualStage(256, 128)
        model.decoder.stages[0] = ScaledDualStage(128, 64)

    return model

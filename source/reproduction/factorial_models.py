"""Composable encoder / skip / bottleneck-decoder ablation model builders."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class FactorialArm:
    family: str
    encoder: bool
    skip: bool
    bottle_decoder: bool
    label: str


ARMS = {
    "a1_swift_only": FactorialArm("A1", True, False, False, "SwiftFormer encoder only"),
    "a1_skip_only": FactorialArm("A1", False, True, False, "Skip group only"),
    "a1_bottle_decoder_only": FactorialArm("A1", False, False, True, "Bottle+D2/D3 group only"),
    "a1s1_tinyvim_skip_only": FactorialArm("A1-S1-SKIPv0.6", True, True, False, "TinyViM+Skip-v0.6"),
    "a1s1_skip_bottle_decoder": FactorialArm("A1-S1-SKIPv0.6", False, True, True, "Skip-v0.6+Bottle+D2/D3"),
    "a1s1_skip_only": FactorialArm("A1-S1-SKIPv0.6", False, True, False, "Skip-v0.6 only"),
    "a1s1_bottle_decoder_only": FactorialArm("A1-S1-SKIPv0.6", False, False, True, "Bottle+D2/D3 only"),
}


class SkipIdentity(nn.Module):
    """Disable the learned skip interaction while preserving its two-tensor API."""

    def forward(self, skip, decoder):
        return skip, decoder


class PlainUNetPyramid(nn.Module):
    """Plain convolutional pyramid aligned to the four-scale A1 decoder API."""

    def __init__(self):
        super().__init__()
        from source.comparison.unet2d import DoubleConv

        self.blocks = nn.ModuleList([
            DoubleConv(3, 32), DoubleConv(32, 64), DoubleConv(64, 128),
            DoubleConv(128, 256), DoubleConv(256, 512), DoubleConv(512, 512),
        ])
        self.pool = nn.MaxPool2d(2)
        self.loading_audit = {
            "encoder": "plain UNet2D convolutional pyramid",
            "pretrained": False,
            "features_before_alignment": "[128@64,256@32,512@16,512@8]",
            "feature_alignment": "trainable 1x1 projections to [64,128,256,512]",
        }

    def forward(self, x):
        features = []
        for index, block in enumerate(self.blocks):
            if index:
                x = self.pool(x)
            x = block(x)
            if index >= 2:
                features.append(x)
        return features


def _replace_encoder_with_plain_unet(model: nn.Module) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        encoder = PlainUNetPyramid()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(43)
        align = nn.ModuleList(
            nn.Conv2d(cin, cout, 1)
            for cin, cout in zip((128, 256, 512, 512), (64, 128, 256, 512))
        )
    model.encoder = encoder
    model.align = align


def _remove_bottle_decoder_group(model: nn.Module) -> None:
    from Models.TinyViM.four_stage_decoder import FourStageDecoder

    decoder = model.decoder.base if hasattr(model.decoder, "base") else model.decoder
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5203)
        plain = FourStageDecoder()
    # The shared A1 decoder calls e2 unconditionally. Identity is exactly the
    # ablation semantics (send the deepest feature directly to D3) and keeps
    # the decoder forward path valid without adding parameters.
    decoder.e2 = nn.Identity()
    decoder.stages[2] = plain.stages[2]
    decoder.stages[3] = plain.stages[3]


def build_factorial_model(arm_name: str, pretrained: bool = True) -> nn.Module:
    arm = ARMS[arm_name]
    if arm.family == "A1":
        from operations.encoder_swap_A016_20261107.encoder_swap_candidate import build_candidate

        model = build_candidate("swiftformer_s__A1", pretrained=pretrained if arm.encoder else False)
    else:
        from source.a1_s1_skip1_v06_20261108.candidate import build_candidate

        model = build_candidate(pretrained=pretrained if arm.encoder else False)

    if not arm.encoder:
        _replace_encoder_with_plain_unet(model)

    decoder = model.decoder.base if hasattr(model.decoder, "base") else model.decoder
    if not arm.skip:
        decoder.e3 = SkipIdentity()
    if not arm.bottle_decoder:
        _remove_bottle_decoder_group(model)

    model.factorial_arm = {
        "family": arm.family,
        "encoder_idea": arm.encoder,
        "skip_group": arm.skip,
        "bottle_d2_d3_group": arm.bottle_decoder,
        "display": arm.label,
    }
    return model

"""Split the grouped A1 Bottle / D2-D3 factorial component into two arms."""
from __future__ import annotations

import torch
from torch import nn

from operations.encoder_swap_A016_20261107.encoder_swap_candidate import (
    build_candidate as build_encoder_swap,
)


ARMS = {
    "E+S+D_no_bottle": {
        "display": "E+S+D (Bottle removed)",
        "description": "SwiftFormer-S + A1 E3 skip + original A1 D2/D3; e2 set to Identity",
    },
    "E+S+B_bottle_only": {
        "display": "E+S+B (D2/D3 plain)",
        "description": "SwiftFormer-S + A1 E3 skip + original A1 E2A; D2/D3 restored to plain U-Net blocks",
    },
}


def build_candidate(arm: str, pretrained: bool = True):
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {tuple(ARMS)}")
    model = build_encoder_swap("swiftformer_s__A1", pretrained=pretrained)
    decoder = model.decoder.base if hasattr(model.decoder, "base") else model.decoder

    if arm == "E+S+D_no_bottle":
        # Retain A1's D-WPO (D3) and DualStage (D2), remove only the E2A Bottle.
        decoder.e2 = nn.Identity()
    else:
        # Retain A1's E2A Bottle, restore only D3 and D2 to the plain decoder.
        from Models.TinyViM.four_stage_decoder import build_four_stage_baseline

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(5213)
            plain = build_four_stage_baseline(pretrained=False)
        decoder.stages[2] = plain.decoder.stages[2]
        decoder.stages[3] = plain.decoder.stages[3]

    model.factorial_arm = {
        "family": "A1",
        "encoder": "SwiftFormer-S",
        "skip": "A1 E3 retained",
        "bottle": arm == "E+S+B_bottle_only",
        "d2_d3": arm == "E+S+D_no_bottle",
        "display": ARMS[arm]["display"],
    }
    return model

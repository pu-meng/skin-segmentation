"""Single decoder-stage additions on top of the E+S+B arm."""
from __future__ import annotations

import torch

from operations.encoder_swap_A016_20261107.encoder_swap_candidate import (
    build_candidate as build_encoder_swap,
)


ARMS = {
    "E+S+B+D3": "retain E2A Bottle and A1 D-WPO/D3; restore D2 to plain",
    "E+S+B+D2": "retain E2A Bottle and A1 DualStage/D2; restore D3 to plain",
}


def build_candidate(arm: str, pretrained: bool = True):
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {tuple(ARMS)}")
    model = build_encoder_swap("swiftformer_s__A1", pretrained=pretrained)
    decoder = model.decoder.base if hasattr(model.decoder, "base") else model.decoder
    a1_d2, a1_d3 = decoder.stages[2], decoder.stages[3]

    from Models.TinyViM.four_stage_decoder import build_four_stage_baseline

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5223)
        plain = build_four_stage_baseline(pretrained=False)
    decoder.stages[2] = plain.decoder.stages[2]
    decoder.stages[3] = plain.decoder.stages[3]

    if arm == "E+S+B+D3":
        decoder.stages[3] = a1_d3
    else:
        decoder.stages[2] = a1_d2

    model.factorial_arm = {
        "family": "A1",
        "encoder": "SwiftFormer-S",
        "skip": "A1 E3 retained",
        "bottle": "A1 E2A retained",
        "d3": arm == "E+S+B+D3",
        "d2": arm == "E+S+B+D2",
        "display": arm,
    }
    return model

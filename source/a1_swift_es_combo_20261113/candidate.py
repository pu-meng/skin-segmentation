"""A1 SwiftFormer E+S arm with Bottle and D2/D3 removed."""
from __future__ import annotations

from torch import nn
from operations.encoder_swap_A016_20261107.encoder_swap_candidate import build_candidate as build_encoder_swap
from source.reproduction.factorial_models import _remove_bottle_decoder_group

ARM = "E+S_no_bottle_no_d2d3"

def build_candidate(arm: str, pretrained: bool = True):
    if arm != ARM:
        raise ValueError(f"unknown arm: {arm}")
    model = build_encoder_swap("swiftformer_s__A1", pretrained=pretrained)
    _remove_bottle_decoder_group(model)
    decoder = model.decoder.base if hasattr(model.decoder, "base") else model.decoder
    assert isinstance(decoder.e2, nn.Identity)
    model.factorial_arm = {
        "family": "A1", "encoder": "SwiftFormer-S", "skip": "A1 E3 retained",
        "bottle": False, "d2_d3": False, "display": "E+S (Bottle and D2/D3 removed)",
    }
    return model

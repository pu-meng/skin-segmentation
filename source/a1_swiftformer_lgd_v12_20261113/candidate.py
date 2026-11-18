"""A1-swiftformer-LGD-v1.2: replace the four A1 decoder stages with LGD."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
for entry in (ROOT, ROOT / "source/comparison", ROOT / "source/bottleneck_mamba"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import torch
from torch import nn

from pretrained_unet2d import prepare_imports


MODEL_NAME = "A1-swiftformer-LGD-v1.2"
NEW_LAYERS_SEED = 20261113


def build_candidate(pretrained: bool = True, encoder: str = "swiftformer_s",
                   mamba_factory: Callable | None = None):
    """Build an independent candidate; preserve SwiftFormer-S and A1 dual E3 skip."""
    if encoder != "swiftformer_s":
        raise ValueError(f"only swiftformer_s is defined, got {encoder!r}")

    # This compatibility shim must precede the original A1 import, whose E2A/D2
    # definitions import the project's actual Mamba-1 implementation.
    prepare_imports("mambavision_t")
    prepare_imports("tinyvim_s")

    from operations.encoder_swap_A016_20261107.encoder_swap_candidate import (
        build_candidate as build_encoder_swap,
    )
    from Models.A1SwiftFormerLGD.model import LGDStage, STAGE_CONFIGS

    model = build_encoder_swap("swiftformer_s__A1", pretrained=pretrained)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(NEW_LAYERS_SEED)
        model.decoder.e2 = nn.Identity()
        # Decoder forward uses indices [3,2,1,0] at D3,D2,D1,D0 respectively.
        model.decoder.stages = nn.ModuleList([
            LGDStage(STAGE_CONFIGS["D0"], mamba_factory=mamba_factory),
            LGDStage(STAGE_CONFIGS["D1"], mamba_factory=mamba_factory),
            LGDStage(STAGE_CONFIGS["D2"], mamba_factory=mamba_factory),
            LGDStage(STAGE_CONFIGS["D3"], mamba_factory=mamba_factory),
        ])

    assert type(model.decoder.e3).__name__ == "E3", (
        "A1's original dual-source E3 skip must remain intact"
    )
    assert len(model.decoder.stages) == 4
    assert [type(stage).__name__ for stage in model.decoder.stages] == ["LGDStage"] * 4
    return model

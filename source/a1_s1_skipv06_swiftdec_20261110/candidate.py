"""A1-S1-SKIPv0.6 with selected decoder stages replaced by official SwiftFormer blocks.

Each replaced stage is `1x1 channel alignment + N official SwiftFormerEncoder blocks`
at that stage's native resolution. D3/D2 (D3S1/D2S1) and both skips stay intact by
default; only the remaining DoubleConv stages (D1, D0) are swapped.
"""
import importlib.util
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.a1_s1_skip1_v06_20261108.candidate import build_candidate as build_parent

SWIFTFORMER_SOURCE = ROOT / 'references/code/SwiftFormer/models/swiftformer.py'
_MODULE_KEY = 'official_swiftformer_decoder_blocks'

# level -> (in_channels at stage entry, out_channels expected by the next stage)
STAGE_SHAPES = {3: (512, 512), 2: (512, 256), 1: (256, 128), 0: (128, 64)}


def official_swiftformer():
    """Load the official SwiftFormer source once, without registering its models."""
    if _MODULE_KEY not in sys.modules:
        if 'timm.models.layers.helpers' not in sys.modules:
            try:
                importlib.import_module('timm.models.layers.helpers')
            except ModuleNotFoundError:
                sys.modules['timm.models.layers.helpers'] = importlib.import_module('timm.layers.helpers')
        spec = importlib.util.spec_from_file_location(_MODULE_KEY, str(SWIFTFORMER_SOURCE))
        module = importlib.util.module_from_spec(spec)
        sys.modules[_MODULE_KEY] = module
        spec.loader.exec_module(module)
    return sys.modules[_MODULE_KEY]


class SwiftDecoderStage(nn.Module):
    """1x1 channel alignment followed by N official SwiftFormerEncoder blocks."""

    def __init__(self, c_in, c_out, depth=1, mlp_ratio=4., layer_scale_init_value=1e-5):
        super().__init__()
        if depth < 1:
            raise ValueError('depth must be >= 1')
        block = official_swiftformer().SwiftFormerEncoder
        self.c_in, self.c_out, self.depth = c_in, c_out, depth
        self.align = nn.Conv2d(c_in, c_out, 1)
        self.blocks = nn.ModuleList([
            block(dim=c_out, mlp_ratio=mlp_ratio, act_layer=nn.GELU,
                  use_layer_scale=True, layer_scale_init_value=layer_scale_init_value)
            for _ in range(depth)
        ])

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.c_in:
            raise ValueError(f'SwiftDecoderStage expects [B,{self.c_in},H,W], got {tuple(x.shape)}')
        x = self.align(x)
        for block in self.blocks:
            x = block(x)
        return x


def build_candidate(pretrained=True, stages=(1, 0), depth=1, mlp_ratio=4., seed_base=4713):
    """Replace the listed decoder stages. stages=(1,0) keeps D3S1/D2S1 and skips."""
    levels = tuple(stages)
    unknown = [level for level in levels if level not in STAGE_SHAPES]
    if unknown:
        raise ValueError(f'unknown decoder levels {unknown}; expected a subset of {sorted(STAGE_SHAPES)}')
    model = build_parent(pretrained=pretrained)
    for level in levels:
        c_in, c_out = STAGE_SHAPES[level]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed_base + level)
            model.decoder.stages[level] = SwiftDecoderStage(c_in, c_out, depth=depth, mlp_ratio=mlp_ratio)
    return model


def replaced_levels(model):
    """Levels whose decoder stage is a SwiftDecoderStage, for audit records."""
    return [level for level in sorted(STAGE_SHAPES, reverse=True)
            if isinstance(model.decoder.stages[level], SwiftDecoderStage)]

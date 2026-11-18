"""A1-S1-SKIPv0.6 with selected decoder stages replaced by official MobileOne blocks.

Each replaced stage is N official MobileOneBlock modules at that stage's native
resolution. MobileOneBlock maps in_channels -> out_channels and takes stride
itself, so no extra 1x1 alignment is needed (unlike the SwiftFormer version).
D3S1/D2S1/BottleneckS1/Skip1V0.6 stay intact; only D1 and D0 are swapped.
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

MOBILEONE_SOURCE = ROOT / 'references/code/MobileOne/mobileone.py'
_MODULE_KEY = 'official_mobileone_decoder_blocks'

# level -> (in_channels at stage entry, out_channels expected by the next stage)
STAGE_SHAPES = {3: (512, 512), 2: (512, 256), 1: (256, 128), 0: (128, 64)}


def official_mobileone():
    """Load the official Apple MobileOne source once."""
    if _MODULE_KEY not in sys.modules:
        spec = importlib.util.spec_from_file_location(_MODULE_KEY, str(MOBILEONE_SOURCE))
        module = importlib.util.module_from_spec(spec)
        sys.modules[_MODULE_KEY] = module
        spec.loader.exec_module(module)
    return sys.modules[_MODULE_KEY]


class MobileOneDecoderStage(nn.Module):
    """N stacked official MobileOneBlock modules at one decoder resolution."""

    def __init__(self, c_in, c_out, depth=1, kernel_size=3, use_se=False, num_conv_branches=1):
        super().__init__()
        if depth < 1:
            raise ValueError('depth must be >= 1')
        if kernel_size % 2 != 1:
            raise ValueError('kernel_size must be odd to keep resolution')
        block = official_mobileone().MobileOneBlock
        self.c_in, self.c_out, self.depth = c_in, c_out, depth
        self.blocks = nn.ModuleList([
            block(in_channels=c_in if i == 0 else c_out,
                  out_channels=c_out,
                  kernel_size=kernel_size,
                  stride=1,
                  padding=kernel_size // 2,
                  groups=1,
                  inference_mode=False,
                  use_se=use_se,
                  num_conv_branches=num_conv_branches)
            for i in range(depth)
        ])

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.c_in:
            raise ValueError(f'MobileOneDecoderStage expects [B,{self.c_in},H,W], got {tuple(x.shape)}')
        for block in self.blocks:
            x = block(x)
        return x


def reparameterize(model):
    """Fuse every MobileOneBlock branch into one conv. Call on a deepcopy, in eval mode."""
    count = 0
    for module in model.modules():
        if isinstance(module, official_mobileone().MobileOneBlock) and not module.inference_mode:
            module.reparameterize()
            count += 1
    return count


def build_candidate(pretrained=True, stages=(1, 0), depth=1, kernel_size=3, use_se=False,
                    num_conv_branches=1, seed_base=4714):
    levels = tuple(stages)
    unknown = [level for level in levels if level not in STAGE_SHAPES]
    if unknown:
        raise ValueError(f'unknown decoder levels {unknown}; expected a subset of {sorted(STAGE_SHAPES)}')
    model = build_parent(pretrained=pretrained)
    for level in levels:
        c_in, c_out = STAGE_SHAPES[level]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed_base + level)
            model.decoder.stages[level] = MobileOneDecoderStage(
                c_in, c_out, depth=depth, kernel_size=kernel_size,
                use_se=use_se, num_conv_branches=num_conv_branches)
    return model


def replaced_levels(model):
    return [level for level in sorted(STAGE_SHAPES, reverse=True)
            if isinstance(model.decoder.stages[level], MobileOneDecoderStage)]

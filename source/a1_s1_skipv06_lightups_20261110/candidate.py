"""A1-S1-SKIPv0.6 with parameter-free upsampling and optional lighter channel alignment.

Two independent switches, both off by default so the parent is reproduced exactly:

  ups='conv'   (default) original three ConvTranspose2d
  ups='interp' bilinear interpolate x2 + 1x1 projection, no learnable spatial kernel

  align='full'     (default) original four 1x1 Conv2d
  align='lowrank'  two chained 1x1 convs through rank r
  align='group8'   grouped 1x1 Conv2d with 8 groups
"""
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.a1_s1_skip1_v06_20261108.candidate import build_candidate as build_parent

# decoder.ups[i] pairs as used by Decoder.forward: (in, out)
UP_SHAPES = {2: (512, 256), 1: (256, 128), 0: (128, 64)}
# TinyViM-S stage channels -> decoder channels
ALIGN_SHAPES = [(48, 64), (64, 128), (168, 256), (224, 512)]


class InterpUp(nn.Module):
    """Bilinear x2 upsample followed by a 1x1 channel projection."""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.proj = nn.Conv2d(c_in, c_out, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.proj(x)


class LowRankAlign(nn.Module):
    """Two 1x1 convs through a rank-r bottleneck."""

    def __init__(self, c_in, c_out, rank):
        super().__init__()
        self.down = nn.Conv2d(c_in, rank, 1)
        self.up = nn.Conv2d(rank, c_out, 1)

    def forward(self, x):
        return self.up(self.down(x))


def build_candidate(pretrained=True, ups='conv', align='full', rank=16, groups=8):
    if ups not in ('conv', 'interp'):
        raise ValueError(f"ups must be 'conv' or 'interp', got {ups!r}")
    if align not in ('full', 'lowrank', 'group8'):
        raise ValueError(f"align must be 'full', 'lowrank' or 'group8', got {align!r}")
    model = build_parent(pretrained=pretrained)
    if ups == 'interp':
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(4720)
            model.decoder.ups = nn.ModuleList(
                InterpUp(*UP_SHAPES[level]) for level in (0, 1, 2))
    if align != 'full':
        modules = []
        with torch.random.fork_rng(devices=[]):
            for index, (c_in, c_out) in enumerate(ALIGN_SHAPES):
                torch.manual_seed(4721 + index)
                if align == 'lowrank':
                    modules.append(LowRankAlign(c_in, c_out, rank))
                else:
                    if c_in % groups or c_out % groups:
                        raise ValueError(f'group8 needs both channels divisible by 8: {c_in}->{c_out}')
                    modules.append(nn.Conv2d(c_in, c_out, 1, groups=groups))
            model.align = nn.ModuleList(modules)
    return model

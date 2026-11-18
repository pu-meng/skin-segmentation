"""Isolated E1.1 variant; preserve the completed E1 implementation and run."""
from pathlib import Path
import importlib.util
import math
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / 'source/three_module_impl_20261104'
sys.path[:0] = [str(ROOT), str(ROOT/'source/comparison'), str(ORIGINAL)]
spec = importlib.util.spec_from_file_location('_completed_e1_candidate', ORIGINAL/'candidate.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
sys.path.insert(0, str(Path(__file__).parent))


def separable_conv(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False),
        nn.Conv2d(cin, cout, 1, bias=False),
        nn.BatchNorm2d(cout), nn.GELU(),
    )


class ConvWaveStage(nn.Module):
    def __init__(self, cin, channels, resolution):
        super().__init__()
        self.pre = separable_conv(cin, channels)
        self.wave = original.DirectionalWaveOperator(
            res=resolution, dim=channels, hidden_dim=channels, num_heads=1,
        )
        self.alpha_embed = nn.Parameter(torch.full(
            (resolution, resolution//2+1, 1), math.log(math.expm1(.05)),
        ))
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), .001))
        self.post = separable_conv(channels, channels)

    def forward(self, x):
        x = self.pre(x)
        return self.post(x + self.gamma*self.wave(x, self.alpha_embed))


def build_candidate(pretrained=True):
    from Models.TinyViM.four_stage_decoder import build_four_stage_baseline
    model = build_four_stage_baseline(pretrained=pretrained)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model.decoder.stages = nn.ModuleList([
            ConvWaveStage(128, 64, 64),
            ConvWaveStage(256, 128, 32),
            ConvWaveStage(512, 256, 16),
            ConvWaveStage(512, 512, 8),
        ])
    return model

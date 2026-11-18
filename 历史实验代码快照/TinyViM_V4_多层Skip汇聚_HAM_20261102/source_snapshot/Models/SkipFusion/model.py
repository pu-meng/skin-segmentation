"""V4: three depthwise-separable skip branches sum into baseline decoder D0."""
from pathlib import Path
import sys
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'source/comparison'))
from pretrained_unet2d import PretrainedUNet2D, prepare_imports


class MultiScaleSkipFusion(nn.Module):
    """S0/S1/S2 -> DW3x3 strides4/2/1 -> PW1x1 -> elementwise sum.

    Bias-free convolutions; no normalization, activation, interpolation or gate.
    """
    def __init__(self, seed=44):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.branches = nn.ModuleList([
                nn.Sequential(nn.Conv2d(c, c, 3, stride=stride, padding=1, groups=c, bias=False),
                              nn.Conv2d(c, 256, 1, bias=False))
                for c, stride in zip((64, 128, 256), (4, 2, 1))
            ])

    def forward(self, skips):
        if len(skips) != 3:
            raise ValueError('Expected S0,S1,S2')
        expected = ((64,64,64), (128,32,32), (256,16,16))
        batch = skips[0].shape[0]
        for x, shape in zip(skips, expected):
            if x.ndim != 4 or x.shape[0] != batch or tuple(x.shape[1:]) != shape:
                raise ValueError(f'Expected shared batch and C,H,W={shape}, got {tuple(x.shape)}')
        outputs = [branch(x) for branch, x in zip(self.branches, skips)]
        return outputs[0] + outputs[1] + outputs[2]


class TinyViMV4SkipFusion(PretrainedUNet2D):
    def __init__(self, pretrained=True):
        prepare_imports('mambavision_t')
        super().__init__('tinyvim_s', pretrained=pretrained, new_layers_seed=42)
        self.skip_fusion = MultiScaleSkipFusion(seed=44)

    def forward(self, x):
        features = [align(f) for align, f in zip(self.align, self.raw_features(x))]
        # Preserve S0->D2, S1->D1, deepest8x8->up0, and the whole shared decoder.
        fused = self.skip_fusion(features[:3])
        return self.decoder([features[0], features[1], fused, features[3]])

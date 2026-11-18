"""A1-S1-SKIPv0.6 (TinyViM-S) ablations: restore ONE block at a time to the plain UNet form.

Base = source/a1_s1_skip1_v06_20261108.build_candidate (8,933,978 params, ISIC no-TTA 91.3015).
That model differs from the plain four-stage UNet decoder in four places:
  e3     : Skip1-v0.6 (encoder-side correction)        -> variant no_skip1: removed, plain concat
  e2     : BottleneckS1                                -> variant no_e2   : removed, f[3] straight in
  stages : stages[3] = D3S1, stages[2] = D2S1          -> variant no_d2d3 : both back to DoubleConv
                                            （DoubleConv from Models.TinyViM.four_stage_decoder baseline）

Only the selected block changes; encoder, alignment, other stages, head, loss and schedule stay
exactly as in the base run so the delta is attributable to one removal.
"""
import torch
from torch import nn
from torch.nn import functional as F

from source.a1_s1_skip1_v06_20261108.candidate import build_candidate as build_parent

VARIANTS = ('none', 'no_skip1', 'no_e2', 'no_d2d3')


class AblDecoder(nn.Module):
    def __init__(self, base, variant):
        super().__init__()
        self.base = base
        self.variant = variant

    def forward(self, f):
        b = self.base
        x = b.stages[3](f[3]) if b.e2 is None else b.stages[3](b.e2(f[3]))
        for level in (2, 1, 0):
            x = b.ups[level](x)
            s = f[level]
            if level == 0 and b.e3 is not None:
                s, x = b.e3(s, x)
            x = b.stages[level](torch.cat([x, s], 1))
        return torch.sigmoid(F.interpolate(b.head(x), size=(256, 256),
                                           mode='bilinear', align_corners=False))


def build_candidate(pretrained=True, variant='none'):
    assert variant in VARIANTS, variant
    model = build_parent(pretrained=pretrained)
    dec = model.decoder
    if variant == 'no_d2d3':
        from Models.TinyViM.four_stage_decoder import build_four_stage_baseline
        base = build_four_stage_baseline(pretrained=False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(4301)
            dec.stages[2] = base.decoder.stages[2]
            dec.stages[3] = base.decoder.stages[3]
    if variant == 'no_skip1':
        dec.e3 = None
    if variant == 'no_e2':
        dec.e2 = None
    model.decoder = AblDecoder(dec, variant)
    return model

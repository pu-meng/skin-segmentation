"""Ablation of A1+SwiftFormer-S: restore one A0 block at a time to the plain UNet form.

Base model = the exact A1 decoder with the official pretrained SwiftFormer-S encoder
(the same construction that scored 91.8683 ISIC2018 no-TTA: build_swap('swiftformer_s__A1')).
A1's decoder differs from the plain four-stage UNet decoder in only three places:
  e3     : dual-source skip (E3) before the level-0 concat
  e2     : bottleneck module (E2) in front of stages[3]
  stages : stages[3] = DStage (directional wave), stages[2] = DualStage (Mamba + RepDW7)

variant='no_skip1' : drop e3      -> level-0 concat uses the encoder feature unchanged
variant='no_e2'    : drop e2      -> f[3] goes straight into stages[3]
variant='no_d2d3'  : stages[2]/[3] replaced by the plain decoder blocks from
                     Models.TinyViM.four_stage_decoder.build_four_stage_baseline

Everything else (encoder, channel alignment, other stages, head, loss, schedule) is untouched.
"""
import torch
from torch import nn
from torch.nn import functional as F

from operations.encoder_swap_A016_20261107.encoder_swap_candidate import build_candidate as build_swap


class DecoderAbl(nn.Module):
    """A1 decoder forward with the selected block restored to the plain form."""

    def __init__(self, old, variant):
        super().__init__()
        self.stages = old.stages
        self.ups = old.ups
        self.head = old.head
        # Restored blocks are removed outright (not merely skipped) so that the parameter
        # count reported for the arm matches what actually runs.
        self.e2 = None if variant == 'no_e2' else old.e2
        self.e3 = None if variant == 'no_skip1' else old.e3
        self.variant = variant

    def forward(self, f):
        if self.variant == 'no_e2':
            x = self.stages[3](f[3])
        else:
            x = self.stages[3](self.e2(f[3]))
        for level in (2, 1, 0):
            x = self.ups[level](x)
            s = f[level]
            if level == 0 and self.variant != 'no_skip1':
                s, x = self.e3(s, x)
            x = self.stages[level](torch.cat([x, s], 1))
        return torch.sigmoid(F.interpolate(self.head(x), size=(256, 256),
                                           mode='bilinear', align_corners=False))


def build_candidate(pretrained=True, encoder='swiftformer_s', variant='none'):
    model = build_swap(f'{encoder}__A1', pretrained=pretrained)
    model.decoder = DecoderAbl(model.decoder, variant)
    if variant == 'no_d2d3':
        from Models.TinyViM.four_stage_decoder import build_four_stage_baseline
        base = build_four_stage_baseline(pretrained=False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(4204)
            model.decoder.stages[2] = base.decoder.stages[2]
            model.decoder.stages[3] = base.decoder.stages[3]
    elif variant in ('none', 'no_skip1', 'no_e2'):
        pass
    else:
        raise ValueError(f'unknown variant {variant}')
    return model

"""Ablation arm 3: Skip2/Skip3 become one ConvNeXt block per side.

Same decoder wiring as the S2S3 candidate (Skip3 at level2, Skip2 at level1, inserted before
concat), but each position now holds only the ContentBlock -- the ConvNeXt-style inverted
bottleneck (DW7 -> BN -> 1x1 C->4C -> GELU -> 1x1 4C->C, residual with learnable beta).
The cross-gate (CrossGate) and the parallel local DW5 branch (LocalConv) are removed, so there
is no D->S / S->D interaction at these two levels; each side is corrected on its own.

This keeps the paper's ablation question clean: arms 1/2 keep one full S2S3 block, arm 3 keeps
the same "strengthen the skip" idea but with the interaction removed.
"""
import torch
from torch import nn

from source.a1_s2s3_20261111.candidate import (  # noqa: F401
    ContentBlock,
    DecoderS2S3,
    SKIP_CHANNELS,
    build_a1,
)


class SkipConvNeXt(nn.Module):
    """s' = ConvNeXtBlock_s(s), d unchanged -- only the encoder side is corrected.

    Same one-sided convention as Skip1-v0.6 (S' = S + gamma*delta); there is no D-side branch and
    no cross interaction, so the ablation isolates "ConvNeXt block on the encoder-side skip".
    """

    def __init__(self, c, expand=4, beta=1.0):
        super().__init__()
        self.c = c
        self.content_s = ContentBlock(c, expand, beta)
        # Compatibility attributes for the shared diagnostic trainer; never used in forward.
        self.runtime = {'enabled': True, 'use_gate': False, 'use_local': False}
        self.collect_stats = False
        self.stats = None

    def forward(self, s, d):
        return self.content_s(s), d


def build_candidate(pretrained=True, use_skip2=True, use_skip3=True, **kwargs):
    model = build_a1(pretrained=pretrained)
    old = model.decoder
    skip2 = skip3 = None
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4202)
        if use_skip3:
            c, expand = SKIP_CHANNELS['skip3']
            skip3 = SkipConvNeXt(c, expand)
        torch.manual_seed(4203)
        if use_skip2:
            c, expand = SKIP_CHANNELS['skip2']
            skip2 = SkipConvNeXt(c, expand)
    model.decoder = DecoderS2S3(old, skip2, skip3)
    return model

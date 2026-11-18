"""Decoder-conditioned attenuation of the raw S0 skip, before concatenation."""
import torch
from torch import nn
from torch.nn import functional as F


class DirectSkipGate(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.proj = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def weights(self, decoder):
        return self.proj(decoder).sigmoid()

    def forward(self, skip, decoder):
        if skip.shape != decoder.shape:
            raise ValueError('Decoder and skip must have identical BCHW shape')
        return skip * self.weights(decoder)


class Decoder(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.stages = original.stages
        self.ups = original.ups
        self.head = original.head
        self.gate = DirectSkipGate(64)
        self.enabled = True

    def forward(self, features):
        x = self.stages[3](features[3])
        for level in (2, 1, 0):
            x = self.ups[level](x)
            skip = features[level]
            if level == 0 and self.enabled:
                skip = self.gate(skip, x)
            x = self.stages[level](torch.cat((x, skip), dim=1))
        logits = F.interpolate(self.head(x), size=(256, 256), mode='bilinear', align_corners=False)
        return logits.sigmoid()


def build_candidate(pretrained=True, enabled=True):
    from Models.TinyViM.four_stage_decoder import build_four_stage_baseline
    model = build_four_stage_baseline(pretrained)
    if enabled:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            model.decoder = Decoder(model.decoder)
    return model

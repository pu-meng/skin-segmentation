"""A1-SwiftFormer-v0.3: fold D3 into the bottleneck and use a DoubleConv there."""
import torch
from torch import nn

from operations.encoder_swap_A016_20261107.encoder_swap_candidate import (
    build_candidate as build_encoder_swap,
)


class ConvNormAct(nn.Sequential):
    """One 3x3 convolution followed by BatchNorm and ReLU."""

    def __init__(self, channels):
        super().__init__(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )


class BottleneckWithD3(nn.Module):
    """Treat E2A and D3 as one deep block; D3 is a two-layer DoubleConv."""

    def __init__(self, e2a):
        super().__init__()
        self.bottleneck = e2a
        self.d3 = nn.Sequential(ConvNormAct(512), ConvNormAct(512))

    def forward(self, x):
        return self.d3(self.bottleneck(x))


def build_candidate(pretrained=True, encoder="swiftformer_s"):
    if encoder != "swiftformer_s":
        raise ValueError(f"only swiftformer_s is defined, got {encoder!r}")
    model = build_encoder_swap("swiftformer_s__A1", pretrained=pretrained)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261115)
        model.decoder.e2 = BottleneckWithD3(model.decoder.e2)
        # D3 now lives inside the E2A bottleneck block; keep the decoder slot as identity.
        model.decoder.stages[3] = nn.Identity()
    return model

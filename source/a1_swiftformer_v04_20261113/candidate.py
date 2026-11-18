"""A1-SwiftFormer-v0.4: put D-WPO in the bottleneck and copy D2 at D3."""
import torch
from torch import nn

from operations.encoder_swap_A016_20261107.encoder_swap_candidate import (
    build_candidate as build_encoder_swap,
)
from source.decoder_dual_a_20261106.candidate import DualStage
from source.three_module_impl_20261104.candidate import DStage


class BottleneckWithDWPO(nn.Module):
    """Group the original E2A and D-WPO into one bottleneck block."""

    def __init__(self, e2a):
        super().__init__()
        self.bottleneck = e2a
        self.dwpo = DStage()

    def forward(self, x):
        return self.dwpo(self.bottleneck(x))


def build_candidate(pretrained=True, encoder="swiftformer_s"):
    if encoder != "swiftformer_s":
        raise ValueError(f"only swiftformer_s is defined, got {encoder!r}")
    model = build_encoder_swap("swiftformer_s__A1", pretrained=pretrained)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261116)
        model.decoder.e2 = BottleneckWithDWPO(model.decoder.e2)
        # D3 copies D2's DualStage design, with the 512-channel D3 output width.
        model.decoder.stages[3] = DualStage(ci=512, co=512)
    return model

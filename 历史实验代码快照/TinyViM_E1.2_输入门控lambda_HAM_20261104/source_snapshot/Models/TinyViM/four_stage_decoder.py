"""Independent four-scale baseline; the existing five-stage decoder is frozen."""
import torch
from torch import nn
from torch.nn import functional as F

from .model import DoubleConv


class FourStageDecoder(nn.Module):
    """D3 at 8px -> D2 at 16px -> D1 at 32px -> D0 at 64px.

    D3 processes the deepest encoder feature once, without self-concatenation.
    D2/D1/D0 fuse the three matching encoder skips. The output head projects
    64 channels to one logit, resizes logits by four, then applies sigmoid.
    """
    def __init__(self):
        super().__init__()
        self.stages = nn.ModuleList([
            DoubleConv(128, 64),
            DoubleConv(256, 128),
            DoubleConv(512, 256),
            DoubleConv(512, 512),
        ])
        self.ups = nn.ModuleList([
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ConvTranspose2d(512, 256, 2, stride=2),
        ])
        self.head = nn.Conv2d(64, 1, 1)

    def forward(self, features):
        expected = [(64, 64, 64), (128, 32, 32), (256, 16, 16), (512, 8, 8)]
        if len(features) != 4 or [tuple(f.shape[1:]) for f in features] != expected:
            raise ValueError('Expected four aligned encoder features: ' + str(expected))
        x = self.stages[3](features[3])
        for level in (2, 1, 0):
            x = self.ups[level](x)
            x = self.stages[level](torch.cat((x, features[level]), dim=1))
        logits = self.head(x)
        logits = F.interpolate(logits, size=(256, 256), mode='bilinear', align_corners=False)
        return torch.sigmoid(logits)


def build_four_stage_baseline(pretrained=True):
    # Reuse the audited official encoder and identical channel alignment.
    from pretrained_unet2d import PretrainedUNet2D, prepare_imports
    prepare_imports('mambavision_t')
    model = PretrainedUNet2D('tinyvim_s', pretrained=pretrained, new_layers_seed=42)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model.decoder = FourStageDecoder()
    return model

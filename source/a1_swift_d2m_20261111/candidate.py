"""A1+SwiftFormer-S with MSDUNet's D2M at skip2 / skip3.

D2M in the MSDUNet paper is the "dual-input dynamic enhancement module". In the official code
(references/code/MSDUNet/models/MSDUNet.py) it is implemented as
  FU(x_up, skip)  -> offsets = offset_net(x_up); out = DeformConv2d(skip, offsets)
  x_skip'         -> SpatialAttention(FU(x_up, x_skip))
i.e. the decoder feature predicts deformable offsets that resample the skip feature, then a
spatial attention reweights it. The two classes below copy the official implementations
(lines 17-60 of MSDUNet.py) so this module does not pull in the vendor file's imports.

Where it goes: A1+SwiftFormer-S has a plain concat at level2 (256 ch) and level1 (128 ch).
This candidate replaces those two concats' skip input with D2M(x_up, skip), using the same
kernel/ dilation schedule as MSDUNet's FU_2 / FU_1 (5x5 pad4 dil2 for the 128-ch level,
3x3 pad2 dil2 for the 64-ch level); here level2 uses 5x5 and level1 uses 3x3.
Everything else (encoder, e2, e3, stages, head) is unchanged from A1+SwiftFormer-S.
"""
import torch
import torchvision
from torch import nn
from torch.nn import functional as F

from operations.encoder_swap_A016_20261107.encoder_swap_candidate import build_candidate as build_swap


class SpatialAttention(nn.Module):
    """Copied from MSDUNet.py (class SpatialAttention)."""

    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x) * residual


class FU(nn.Module):
    """Copied from MSDUNet.py (class FU) -- this is the D2M defect-free core."""

    def __init__(self, in_channels, groups, kernel_size=(3, 3), padding=1, stride=1, dilation=1):
        super().__init__()
        self.offset_net = nn.Conv2d(in_channels, 2 * kernel_size[0] * kernel_size[1],
                                    kernel_size=kernel_size, padding=padding, stride=stride,
                                    dilation=dilation, bias=True)
        self.deform_conv = torchvision.ops.DeformConv2d(in_channels, in_channels,
                                                       kernel_size=kernel_size, padding=padding,
                                                       groups=groups, stride=stride,
                                                       dilation=dilation, bias=False)

    def forward(self, x, skip):
        offsets = self.offset_net(x)
        return self.deform_conv(skip, offsets)


class D2M(nn.Module):
    """FU + SpatialAttention, applied on (skip, decoder feature) and returning the new skip."""

    def __init__(self, c, kernel=7, padding=6, dilation=2):
        super().__init__()
        self.fu = FU(c, c, kernel_size=(kernel, kernel), padding=padding, dilation=dilation)
        self.sa = SpatialAttention()

    def forward(self, s, d):
        return self.sa(self.fu(d, s))


class DecoderD2M(nn.Module):
    def __init__(self, old):
        super().__init__()
        self.stages = old.stages
        self.ups = old.ups
        self.head = old.head
        self.e2 = old.e2
        self.e3 = old.e3
        # level2: ups[2] outputs 256 ch and f[2] is 256 ch; level1: 128 / 128.
        self.skip3 = D2M(256, kernel=5, padding=4, dilation=2)
        self.skip2 = D2M(128, kernel=3, padding=2, dilation=2)

    def forward(self, f):
        x = self.stages[3](self.e2(f[3]))
        for level in (2, 1, 0):
            x = self.ups[level](x)
            s = f[level]
            if level == 2:
                s = self.skip3(s, x)
            elif level == 1:
                s = self.skip2(s, x)
            else:
                s, x = self.e3(s, x)
            x = self.stages[level](torch.cat([x, s], 1))
        return torch.sigmoid(F.interpolate(self.head(x), size=(256, 256),
                                           mode='bilinear', align_corners=False))


def build_candidate(pretrained=True, encoder='swiftformer_s', variant='none'):
    if variant not in ('none', 'd2m_skips'):
        raise ValueError(variant)
    model = build_swap(f'{encoder}__A1', pretrained=pretrained)
    if variant == 'd2m_skips':
        model.decoder = DecoderD2M(model.decoder)
    return model

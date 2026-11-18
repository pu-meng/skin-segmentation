"""A1+SwiftFormer-S with MSDUNet's decoder (paper structure, channels fitted to our features).

Scheme A agreed with the user: copy MSDUNet's decoder structure, but set its widths from our own
feature pyramid (64/128/256/512) instead of MSDUNet's [64,128,320,512]. No projection layers are
inserted; the vendor decoder blocks themselves are reused verbatim (MSDUNet.py classes
LayerScale / DeformConv / MultiScaleDeformConv_3x3 / MSDCDecoder_3x3_LS_up / up_layer, and the
FU + SpatialAttention pair that forms D2M).

Wiring (our pyramid: f0 64@256, f1 128@128, f2 256@64, f3 512@32):
  x    = e2(f3)                     512 @ 32
  x_up = up_last(x)                 256 @ 64      (2x)
  s2   = SA(FU_3(x_up, f2))         256 @ 64
  d3   = msecoder_3(cat(x_up, s2))  128 @ 128     (2x)
  s1   = SA(FU_2(d3, f1))           128 @ 128
  d2   = msecoder_2(cat(d3, s1))     64 @ 256     (2x)
  s0   = SA(FU_1(d2, f0))            64 @ 256
  d1   = msecoder_1(cat(d2, s0))     64 @ 256     (no up: we already are at full resolution)
  out  = sigmoid(1x1(d1))

The only deviation from the vendor forward is the last block's upsampling: MSDUNet runs a 512
input and still needs one more 2x before its head; our 256 input is already at full resolution,
so msecoder_1 keeps its structure but skips the up_layer.
"""
import torch
import torchvision
from torch import nn
from torch.nn import functional as F

from operations.encoder_swap_A016_20261107.encoder_swap_candidate import build_candidate as build_swap
from source.a1_swift_d2m_20261111.candidate import FU, SpatialAttention


class LayerScale(nn.Module):
    """Copied from MSDUNet.py."""

    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, 1, 1, 1) * init_value, requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)

    def forward(self, x):
        return F.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])


class DeformConv(nn.Module):
    """Copied from MSDUNet.py."""

    def __init__(self, in_channels, groups, kernel_size=(3, 3), padding=1, stride=1, dilation=1):
        super().__init__()
        self.offset_net = nn.Conv2d(in_channels, 2 * kernel_size[0] * kernel_size[1],
                                    kernel_size=kernel_size, padding=padding, stride=stride,
                                    dilation=dilation, bias=True)
        self.deform_conv = torchvision.ops.DeformConv2d(in_channels, in_channels,
                                                       kernel_size=kernel_size, padding=padding,
                                                       groups=groups, stride=stride,
                                                       dilation=dilation, bias=False)

    def forward(self, x):
        return self.deform_conv(x, self.offset_net(x))


class MultiScaleDeformConv_3x3(nn.Module):
    """Copied from MSDUNet.py: channel split into 4, deformable convs with dilation 1/2/3."""

    def __init__(self, in_channels):
        super().__init__()
        sub = in_channels // 4
        self.deform_conv1 = nn.Conv2d(sub, sub, kernel_size=(1, 1))
        self.deform_conv3 = DeformConv(sub, sub, kernel_size=(3, 3), padding=1, dilation=1)
        self.deform_conv5 = DeformConv(sub, sub, kernel_size=(3, 3), padding=2, dilation=2)
        self.deform_conv7 = DeformConv(sub, sub, kernel_size=(3, 3), padding=3, dilation=3)

    def forward(self, x):
        c = torch.chunk(x, 4, dim=1)
        return torch.cat([self.deform_conv1(c[0]), self.deform_conv3(c[1]),
                          self.deform_conv5(c[2]), self.deform_conv7(c[3])], dim=1)


class UpLayer(nn.Module):
    """Copied from MSDUNet.py (class up_layer)."""

    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2),
                                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                                nn.BatchNorm2d(ch_out), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.up(x)


class MSDCDecoder(nn.Module):
    """Copied from MSDUNet.py (class MSDCDecoder_3x3_LS_up), with an optional final upsampling."""

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 need_up=True, layer_scale_init_value=1e-5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=False),
                                 nn.GELU(), nn.BatchNorm2d(hidden_features))
        self.dwconv = MultiScaleDeformConv_3x3(hidden_features)
        self.act = nn.GELU()
        self.norm = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Sequential(nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=False),
                                 nn.BatchNorm2d(out_features))
        self.norm1 = nn.BatchNorm2d(in_features)
        self.up_conv = UpLayer(out_features, out_features) if need_up else nn.Identity()
        self.layer_scale = (LayerScale(out_features, layer_scale_init_value)
                            if layer_scale_init_value is not None else nn.Identity())

    def forward(self, x_up, x_skip):
        x = self.norm1(torch.cat([x_up, x_skip], dim=1))
        x = self.fc1(x)
        x = self.dwconv(x) + x
        x = self.norm(self.act(x))
        x = self.fc2(x)
        x = self.up_conv(x)
        return self.layer_scale(x) + x


class DecoderMSD(nn.Module):
    def __init__(self, old):
        super().__init__()
        self.e2 = old.e2                      # bottleneck kept: this arm only swaps the decoder
        self.up_last = nn.Sequential(nn.Upsample(scale_factor=2),
                                     nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=True),
                                     nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.msecoder_3 = MSDCDecoder(256 * 2, 256 * 2, 128)
        self.msecoder_2 = MSDCDecoder(128 * 2, 128 * 2, 64)
        self.msecoder_1 = MSDCDecoder(64 * 2, 64 * 2, 64, need_up=False)
        self.FU_3 = FU(256, 256, kernel_size=(5, 5), padding=4, dilation=2)
        self.FU_2 = FU(128, 128, kernel_size=(3, 3), padding=2, dilation=2)
        self.FU_1 = FU(64, 64, kernel_size=(3, 3), padding=2, dilation=2)
        self.FU_E = SpatialAttention()
        self.head = nn.Conv2d(64, 1, kernel_size=1, bias=False)

    def forward(self, f):
        x = self.e2(f[3])
        x_up = self.up_last(x)
        s2 = self.FU_E(self.FU_3(x_up, f[2]))
        d3 = self.msecoder_3(x_up, s2)
        s1 = self.FU_E(self.FU_2(d3, f[1]))
        d2 = self.msecoder_2(d3, s1)
        s0 = self.FU_E(self.FU_1(d2, f[0]))
        d1 = self.msecoder_1(d2, s0)
        return torch.sigmoid(F.interpolate(self.head(d1), size=(256, 256),
                                           mode='bilinear', align_corners=False))


def build_candidate(pretrained=True, encoder='swiftformer_s', variant='none'):
    if variant not in ('none', 'msddec'):
        raise ValueError(variant)
    model = build_swap(f'{encoder}__A1', pretrained=pretrained)
    if variant == 'msddec':
        model.decoder = DecoderMSD(model.decoder)
    return model

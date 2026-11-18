"""Model factory + loss for the two lightweight skip-connection baselines.

CMUNeXt  : FengheTan9/CMUNeXt, ISBI 2024 oral. 3.149M params, 14.46 GFLOPs @256.
FuseUNet : nayutayuki/FuseUNet. nnUNet-2D trunk (6 stages, 32..320) whose decoder is
           replaced by the ODE-based MyDecoder. 4.687M params, 7.94 GFLOPs @256.
           The vendor MyDecoder unpacked a fixed 5-D shape and crashed on 2-D input;
           local 2-D adaptation applied in unet_decoder.py (see .orig-20261111 backup).

Both return raw logits [B,1,256,256]; sigmoid is applied at evaluation.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path('/home/PuMengYu/experiments/skin_segmentation')
CMUNEXT_DIR = ROOT / 'references/code/CMUNeXt'
FUSE_DIR = ROOT / 'references/code/FuseUNet/FuseUNet'

MODELS = ('cmunext', 'fuseunet')


class BceDiceLoss(nn.Module):
    """Same loss the A-series uses: wb*BCEWithLogits + wd*(1 - soft Dice)."""

    def __init__(self, wb=1.0, wd=1.0, smooth=1.0):
        super().__init__()
        self.wb, self.wd, self.smooth = wb, wd, smooth

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        prob = torch.sigmoid(logits)
        dims = tuple(range(1, prob.ndim))
        inter = (prob * target).sum(dims)
        dice = 1.0 - (2 * inter + self.smooth) / (prob.sum(dims) + target.sum(dims) + self.smooth)
        return self.wb * bce + self.wd * dice.mean()


def build_cmunext(pretrained=True):
    if str(CMUNEXT_DIR) not in sys.path:
        sys.path.insert(0, str(CMUNEXT_DIR))
    from network.CMUNeXt import CMUNeXt
    return CMUNeXt(input_channel=3, num_classes=1)


def build_fuseunet(pretrained=True):
    if str(FUSE_DIR) not in sys.path:
        sys.path.insert(0, str(FUSE_DIR))
    from nnunetv2.nets.unet import PlainConvUNet
    return PlainConvUNet(
        input_channels=3,
        n_stages=6,
        features_per_stage=(32, 64, 128, 256, 320, 320),
        conv_op=nn.Conv2d,
        kernel_sizes=(3,) * 6,
        strides=(1, 2, 2, 2, 2, 2),
        n_conv_per_stage=(2,) * 6,
        num_classes=1,
        n_conv_per_stage_decoder=(2,) * 5,
        conv_bias=True,
        norm_op=nn.InstanceNorm2d,
        norm_op_kwargs={'eps': 1e-5, 'affine': True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={'inplace': True, 'negative_slope': 0.01},
        deep_supervision=False,
    )


def build_model(name, pretrained=True):
    if name == 'cmunext':
        return build_cmunext(pretrained)
    if name == 'fuseunet':
        return build_fuseunet(pretrained)
    raise ValueError(name)


@torch.no_grad()
def dice_score(logits, target, threshold=0.5):
    """Per-batch mean Dice at 0.5 threshold (for validation monitoring only)."""
    pred = torch.sigmoid(logits) >= threshold
    gt = target >= 0.5
    dims = tuple(range(1, pred.ndim))
    inter = (pred & gt).sum(dims).float()
    denom = pred.sum(dims).float() + gt.sum(dims).float()
    return float(((2 * inter + 1e-6) / (denom + 1e-6)).mean())

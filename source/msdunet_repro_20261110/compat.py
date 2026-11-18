"""Build the official MSDUNet model without installing mmcv/thop/ptflops.

The vendor code imports mmcv for ConvModule/build_norm_layer/build_activation_layer and
load_checkpoint, thop/ptflops for FLOPs in __main__, and hardcodes a relative checkpoint
path. All four are replaced here; the parameter count is identical because ConvModule is
reimplemented as the same conv+norm+act stack.
"""
import os
import sys
import types

import torch
import torch.nn as nn

MSDUNET_ROOT = '/home/PuMengYu/experiments/skin_segmentation/references/code/MSDUNet'
CKPT_REL = 'pretrained_pth/transxnet/transx-s.pth.tar'


class ConvModule(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=None, norm_cfg=None, act_cfg=None, **kw):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding,
                              dilation=dilation, groups=groups,
                              bias=(bias if bias is not None else norm_cfg is None))
        self.bn = nn.BatchNorm2d(out_ch) if norm_cfg else nn.Identity()
        kind = act_cfg.get('type') if isinstance(act_cfg, dict) else None
        self.act = nn.GELU() if kind == 'GELU' else (nn.ReLU() if kind == 'ReLU' else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


def build_norm_layer(cfg, num_features, postfix=''):
    if cfg is None:
        return ('identity', nn.Identity())
    kind = cfg.get('type')
    if kind in ('BN2d', 'BN', 'SyncBN'):
        return ('bn' + postfix, nn.BatchNorm2d(num_features))
    if kind == 'GN':
        return ('gn' + postfix, nn.GroupNorm(cfg.get('num_groups', 1), num_features))
    if kind == 'LN':
        return ('ln' + postfix, nn.LayerNorm(num_features))
    raise ValueError(f'unsupported norm cfg {cfg}')


def build_activation_layer(cfg=None, **kw):
    kind = cfg.get('type') if isinstance(cfg, dict) else None
    return nn.GELU() if kind == 'GELU' else (nn.ReLU() if kind == 'ReLU' else nn.Identity())


def _module(name, attrs=None):
    module = types.ModuleType(name)
    module.__path__ = []
    for key, value in (attrs or {}).items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def install_stubs():
    mmcv = _module('mmcv')
    cnn = _module('mmcv.cnn', {'build_activation_layer': build_activation_layer})
    bricks = _module('mmcv.cnn.bricks', {
        'ConvModule': ConvModule,
        'build_activation_layer': build_activation_layer,
        'build_norm_layer': build_norm_layer})
    runner = _module('mmcv.runner')
    checkpoint = _module('mmcv.runner.checkpoint', {'load_checkpoint': lambda *a, **k: None})
    mmcv.cnn, mmcv.runner = cnn, runner
    cnn.bricks = bricks
    runner.checkpoint = checkpoint
    _module('thop', {'profile': lambda *a, **k: (0, 0)})
    _module('ptflops', {'get_model_complexity_info': lambda *a, **k: (0, 0)})


def build_msdunet(num_classes=1, weights=None):
    """weights: path to a segmentation checkpoint, or None for ImageNet-only init."""
    install_stubs()
    if MSDUNET_ROOT not in sys.path:
        sys.path.insert(0, MSDUNET_ROOT)
    previous = os.getcwd()
    os.chdir(MSDUNET_ROOT)  # the constructor loads CKPT_REL relative to cwd
    try:
        checkpoint = os.path.join(MSDUNET_ROOT, CKPT_REL)
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f'TransXNet ImageNet weights missing: {checkpoint}')
        from models.MSDUNet import MSDUFormer_Xnet_3463_FUESMS as MSDUNet
        model = MSDUNet(num_classes=num_classes)
    finally:
        os.chdir(previous)
    if weights:
        state = torch.load(weights, map_location='cpu')
        model.load_state_dict(state, strict=True)
    return model

"""Same four-stage segmentation shell; homogeneous spatial mixers in each arm."""
import importlib
import math
import sys
from pathlib import Path
import sys as _project_sys
_PROJECT = Path(__file__).resolve().parents[2]
_project_sys.path.insert(0, str(_PROJECT / 'source/comparison'))

import torch
from torch import nn
from torch.nn import functional as F

ROOT = _PROJECT / 'source/comparison'
sys.path.insert(0, str(ROOT))
from pretrained_unet2d import SharedDecoder, prepare_imports

WIDTHS = (48, 64, 168, 224)
DEPTHS = (3, 3, 9, 6)
KINDS = ('conv', 'transformer', 'mamba')


class ChannelNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class ConvMixer(nn.Sequential):
    def __init__(self, channels):
        super().__init__(nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                         nn.GELU(), nn.Conv2d(channels, channels, 1))


class AttentionMixer(nn.Module):
    """Global four-head attention, no local-convolution branch or window partition."""
    def __init__(self, channels):
        super().__init__()
        self.heads = 4
        self.qkv = nn.Linear(channels, 3 * channels)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x):
        b, c, h, w = x.shape
        z = x.flatten(2).transpose(1, 2)
        q, k, v = self.qkv(z).reshape(b, h*w, 3, 4, c//4).permute(2, 0, 3, 1, 4)
        z = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        z = self.proj(z.transpose(1, 2).reshape(b, h*w, c))
        return z.transpose(1, 2).reshape(b, c, h, w)


class MambaMixer(nn.Module):
    """Full-resolution four-direction selective scan, no spatial conv or frequency split.

    Reuses the real locally installed scan/autograd; this is a custom SS2D mixer,
    not a claimed reproduction of canonical 1-D Mamba or the complete TinyViM block.
    """
    def __init__(self, channels, d_state=8):
        super().__init__()
        prepare_imports('tinyvim_s')
        native = importlib.import_module('model.tvimblock')
        self.scan = native.cross_selective_scan
        rank = math.ceil(channels / 16)
        self.in_proj = nn.Conv2d(channels, channels, 1)
        self.x_proj_weight = nn.Parameter(torch.stack([
            nn.Linear(channels, rank + 2*d_state, bias=False).weight for _ in range(4)]))
        dts = [native.SS2D.dt_init(rank, channels) for _ in range(4)]
        self.dt_projs_weight = nn.Parameter(torch.stack([m.weight for m in dts]))
        self.dt_projs_bias = nn.Parameter(torch.stack([m.bias for m in dts]))
        self.A_logs = native.SS2D.A_log_init(d_state, channels, copies=4, merge=True)
        self.Ds = native.SS2D.D_init(channels, copies=4, merge=True)
        self.out_norm = nn.LayerNorm(channels)
        self.out_proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        z = self.in_proj(x).contiguous()
        z = self.scan(z, self.x_proj_weight, None, self.dt_projs_weight,
                      self.dt_projs_bias, self.A_logs, self.Ds, self.out_norm,
                      nrows=1, delta_softplus=True, force_fp32=True)
        return self.out_proj(z.permute(0, 3, 1, 2).contiguous())


class Block(nn.Module):
    def __init__(self, channels, kind, seed):
        super().__init__()
        # Every non-mixer parameter is bitwise identical between arms at creation.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.norm1 = ChannelNorm(channels)
            self.norm2 = ChannelNorm(channels)
            self.ffn = nn.Sequential(nn.Conv2d(channels, channels*4, 1), nn.GELU(),
                                     nn.Conv2d(channels*4, channels, 1))
            torch.manual_seed(seed + 100000)
            self.mixer = {'conv': ConvMixer, 'transformer': AttentionMixer,
                          'mamba': MambaMixer}[kind](channels)

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        return x + self.ffn(self.norm2(x))


def conv_bn(cin, cout, stride):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.GELU())


class PureMixerUNet(nn.Module):
    def __init__(self, kind, seed=42):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(kind)
        self.kind = kind
        self.seed = seed
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.stem = nn.Sequential(conv_bn(3, 24, 2), conv_bn(24, 48, 2))
            self.down = nn.ModuleList([conv_bn(a, b, 2) for a, b in zip(WIDTHS[:-1], WIDTHS[1:])])
            self.align = nn.ModuleList([nn.Conv2d(a, b, 1) for a, b in zip(WIDTHS, (64,128,256,512))])
            self.decoder = SharedDecoder()
            self.stages = nn.ModuleList([nn.Sequential(*[
                Block(c, kind, seed + 1000 + stage*100 + i) for i in range(n)])
                for stage, (c, n) in enumerate(zip(WIDTHS, DEPTHS))])
        self.register_buffer('mean', torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('std', torch.tensor([.229,.224,.225])[None,:,None,None])

    def encode(self, x):
        if x.shape[1:] != (3,256,256):
            raise ValueError('Expected RGB /255 input [B,3,256,256]')
        x = self.stem((x-self.mean)/self.std)
        features = []
        for i, stage in enumerate(self.stages):
            if i:
                x = self.down[i-1](x)
            x = stage(x)
            features.append(x)
        return features

    def forward(self, x):
        return self.decoder([a(f) for a, f in zip(self.align, self.encode(x))])

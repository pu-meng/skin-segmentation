"""Identical frozen U-Net features; exchange only the probe's token mixer."""
import sys
from pathlib import Path
import sys as _project_sys
_PROJECT = Path(__file__).resolve().parents[3]
_project_sys.path.insert(0, str(_PROJECT / 'source/comparison'))
import torch
from torch import nn

ROOT = _PROJECT / 'source/comparison'
sys.path.insert(0, str(ROOT))
from unet2d import UNet2D


class ConvMixer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
                                 nn.Conv2d(dim, dim, 1))

    def forward(self, tokens):
        b, n, c = tokens.shape
        h = int(n**0.5)
        assert h*h == n
        x = tokens.transpose(1, 2).reshape(b, c, h, h)
        return self.net(x).flatten(2).transpose(1, 2)


class AttentionMixer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, 4, dropout=0, batch_first=True)

    def forward(self, x):
        return self.attn(x, x, x, need_weights=False)[0]


class Probe(nn.Module):
    def __init__(self, kind, seed=42, width=128):
        super().__init__()
        self.kind = kind
        self.width = width
        # Shared fixed projections make mixer inputs exactly equal across groups,
        # including after optimization. BatchNorm never changes in the backbone.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261025)
            self.down = nn.Conv2d(512, width, 1, bias=False)
            self.up = nn.Conv2d(width, 512, 1, bias=False)
            self.norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
            torch.manual_seed(seed+10000)
            self.ffn = nn.Sequential(nn.Linear(width, 2*width), nn.GELU(),
                                     nn.Linear(2*width, width))
            torch.manual_seed(seed)
            if kind == 'conv':
                self.mixer = ConvMixer(width)
            elif kind == 'attention':
                self.mixer = AttentionMixer(width)
            elif kind == 'mamba':
                from pretrained_unet2d import prepare_imports
                prepare_imports('mambavision_t')
                from mamba_ssm.modules.mamba_simple import Mamba
                self.mixer = Mamba(d_model=width, d_state=16, d_conv=4, expand=2)
            else:
                raise ValueError(kind)
        for module in (self.down, self.up):
            module.requires_grad_(False)

    def forward(self, x, capture=False):
        b, _, h, w = x.shape
        t = self.norm(self.down(x).flatten(2).transpose(1, 2))
        mixed = self.mixer(t)
        z = t + mixed
        z = z + self.ffn(self.norm2(z))
        branch = self.up((z-t).transpose(1, 2).reshape(b, self.width, h, w))
        y = x + 0.1*branch
        if capture:
            return y, dict(raw_input=x, mixer_input=t.transpose(1, 2).reshape(b,self.width,h,w),
                           mixer_output=mixed.transpose(1,2).reshape(b,self.width,h,w),
                           branch=branch, residual_output=y)
        return y


class FrozenUNet(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        self.base = UNet2D()
        self.base.load_state_dict(torch.load(checkpoint, map_location='cpu'), strict=True)
        self.base.requires_grad_(False)
        self.base.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def encode(self, x):
        features=[]
        for i, module in enumerate(self.base.encoders):
            if i:
                x=self.base.pool(x)
            x=module(x)
            features.append(x)
        return features

    def decode(self, x, skips):
        for up, decoder, skip in zip(self.base.ups,self.base.decoders,reversed(skips[:-1])):
            x=decoder(torch.cat((up(x),skip),dim=1))
        return torch.sigmoid(self.base.head(x))


def loss_fn(pred, target):
    p=pred.flatten(1);t=target.flatten(1)
    dice=(2*(p*t).sum(1)+1)/(p.sum(1)+t.sum(1)+1)
    return torch.nn.functional.binary_cross_entropy(pred,target)+1-dice.mean()

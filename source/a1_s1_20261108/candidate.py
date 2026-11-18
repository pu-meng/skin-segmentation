"""A1-S1: compact bottleneck, RMT-style D3, and narrow three-layer DSC D2."""
import math
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'source/three_module_impl_20261104'))
from source.three_module_impl_20261104.candidate import DirectionalWaveOperator
from mamba_ssm.modules.mamba_simple import Mamba
from source.decoder_dual_a_20261106.candidate import build_candidate as build_a1


def pw(ci, co):
    return nn.Conv2d(ci, co, 1, bias=True)


def dw(c, k):
    return nn.Conv2d(c, c, k, padding=k // 2, groups=c, bias=False)


class CPE(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dw = dw(c, 3)

    def forward(self, x):
        return x + self.dw(x)


class MambaBranch(nn.Module):
    def __init__(self, c=64):
        super().__init__()
        self.norm = nn.LayerNorm(c)
        self.mixer = Mamba(d_model=c, d_state=16, d_conv=4, expand=2)

    def forward(self, x):
        b, c, h, w = x.shape
        t = x.flatten(2).transpose(1, 2)
        t = t + self.mixer(self.norm(t))
        return t.transpose(1, 2).reshape(b, c, h, w)


class LocalBranch(nn.Module):
    def __init__(self, c, kernel):
        super().__init__()
        self.dw = dw(c, kernel)
        self.norm = nn.GroupNorm(8, c)
        self.act = nn.GELU()
        self.pw = pw(c, c)

    def forward(self, x):
        return self.pw(self.act(self.norm(self.dw(x))))


class BottleneckS1(nn.Module):
    def __init__(self):
        super().__init__()
        self.cpe = CPE(512)
        self.reduce = pw(512, 256)
        self.mamba = MambaBranch(64)
        self.local3 = LocalBranch(96, 3)
        self.local7 = LocalBranch(96, 7)
        self.fuse = pw(256, 256)
        self.ffn_in = pw(256, 128)
        self.ffn_norm = nn.GroupNorm(8, 128)
        self.ffn_act = nn.GELU()
        self.ffn_out = pw(128, 256)
        self.output = pw(256, 512)

    def forward(self, x):
        residual = self.cpe(x)
        m, a, b = self.reduce(residual).split((64, 96, 96), dim=1)
        z = self.fuse(torch.cat((self.mamba(m), self.local3(a), self.local7(b)), dim=1))
        z = self.ffn_out(self.ffn_act(self.ffn_norm(self.ffn_in(z))))
        return residual + self.output(z)


class D3S1(nn.Module):
    def __init__(self):
        super().__init__()
        self.cpe = CPE(512)
        self.wpo = DirectionalWaveOperator(res=8, dim=512, hidden_dim=512, num_heads=1)
        self.alpha_embed = nn.Parameter(torch.full((8, 5, 1), math.log(math.expm1(.05))))
        self.lce = dw(512, 5)
        self.wave_scale = nn.Parameter(torch.full((1, 512, 1, 1), 1e-3))
        self.ffn_pw_in = pw(512, 256)
        self.ffn_norm = nn.GroupNorm(8, 256)
        self.ffn_act1 = nn.GELU()
        self.ffn_dw = dw(256, 3)
        self.ffn_act2 = nn.GELU()
        self.ffn_pw_out = pw(256, 512)
        self.ffn_scale = nn.Parameter(torch.full((1, 512, 1, 1), 1e-3))

    def forward(self, x):
        x = self.cpe(x)
        x = x + self.wave_scale * (self.wpo(x, self.alpha_embed) + self.lce(x))
        f = self.ffn_pw_in(x)
        f = self.ffn_norm(f)
        f = self.ffn_act1(f)
        f = self.ffn_act2(self.ffn_dw(f))
        return x + self.ffn_scale * self.ffn_pw_out(f)


class D2S1(nn.Module):
    """Input is the 512-channel concatenation of decoder and encoder skip."""
    def __init__(self):
        super().__init__()
        self.fusion = pw(512, 256)
        self.reduce = pw(256, 128)
        self.spatial = nn.ModuleList([
            nn.Sequential(dw(128, 3), nn.GroupNorm(8, 128), nn.GELU())
            for _ in range(3)
        ])
        self.expand = pw(128, 256)
        self.scale = nn.Parameter(torch.full((1, 256, 1, 1), 1e-3))

    def forward(self, x):
        u = self.fusion(x)
        z = self.reduce(u)
        for block in self.spatial:
            z = block(z)
        return u + self.scale * self.expand(z)


def build_candidate(pretrained=True):
    model = build_a1(pretrained=pretrained)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4101)
        model.decoder.e2 = BottleneckS1()
        torch.manual_seed(4102)
        model.decoder.stages[3] = D3S1()
        torch.manual_seed(4103)
        model.decoder.stages[2] = D2S1()
    return model

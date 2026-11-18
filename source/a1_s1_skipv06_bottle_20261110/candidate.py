"""A1-S1-SkipV0.6-Bottle: A-M5 V3.1 replaces decoder.e2 only."""
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from mamba_ssm.modules.mamba_simple import Mamba
from source.a1_s1_20261108.candidate import CPE
from source.a1_s1_skip1_v06_20261108.candidate import build_candidate as build_parent


@dataclass(frozen=True)
class BottleConfig:
    enable_mamba: bool = True
    enable_global_query: bool = True
    enable_dwpo: bool = True
    enable_input_repdw: bool = True
    enable_output_repdw: bool = True
    enable_stage4: bool = True
    directional: bool = True
    fixed_t: float = 1.0
    init_speed: float = 1.0


class RepDW(nn.Module):
    def __init__(self, channels, kernel):
        super().__init__()
        self.channels, self.kernel = channels, kernel
        self.spatial = nn.Conv2d(channels, channels, kernel, padding=kernel // 2, groups=channels, bias=False)
        self.spatial_bn = nn.BatchNorm2d(channels)
        self.point = nn.Conv2d(channels, channels, 1, groups=channels, bias=False)
        self.point_bn = nn.BatchNorm2d(channels)
        self.identity_bn = nn.BatchNorm2d(channels)
        self.deploy = None

    def forward(self, x):
        if self.deploy is not None:
            return self.deploy(x)
        return self.spatial_bn(self.spatial(x)) + self.point_bn(self.point(x)) + self.identity_bn(x)

    @staticmethod
    def _fuse(weight, bn):
        scale = bn.weight * torch.rsqrt(bn.running_var + bn.eps)
        return weight * scale[:, None, None, None], bn.bias - bn.running_mean * scale

    @torch.no_grad()
    def switch_to_deploy(self):
        if self.deploy is not None:
            return
        if self.training:
            raise RuntimeError('RepDW fusion requires eval mode')
        w, b = self._fuse(self.spatial.weight, self.spatial_bn)
        p, pb = self._fuse(self.point.weight, self.point_bn)
        identity = torch.zeros_like(w)
        identity[:, 0, self.kernel // 2, self.kernel // 2] = 1
        iw, ib = self._fuse(identity, self.identity_bn)
        pad = self.kernel // 2
        self.deploy = nn.Conv2d(self.channels, self.channels, self.kernel, padding=pad,
                                groups=self.channels, bias=True, device=w.device, dtype=w.dtype)
        self.deploy.weight.copy_(w + F.pad(p, (pad, pad, pad, pad)) + iw)
        self.deploy.bias.copy_(b + pb + ib)
        self.spatial = self.spatial_bn = self.point = self.point_bn = self.identity_bn = None


def local(c, kernel=3):
    return nn.Sequential(RepDW(c, kernel), nn.GroupNorm(8, c), nn.GELU(), nn.Conv2d(c, c, 1))


class ChannelGate(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(c, c // 4), nn.GELU(), nn.Linear(c // 4, c), nn.Sigmoid())

    def forward(self, x):
        return self.net(x.mean((2, 3)))[:, :, None, None]


class DirectionalWPO(nn.Module):
    """WaveR cosine DCT propagation core with per-channel direction; not full WaveR."""
    def __init__(self, channels=128, size=8, fixed_t=1.0, init_speed=1.0, directional=True):
        super().__init__()
        if fixed_t <= 0 or init_speed <= 0:
            raise ValueError('fixed_t and init_speed must be positive')
        n = torch.arange(size, dtype=torch.float32)
        basis = torch.cos(math.pi * n[:, None] * (n[None, :] + .5) / size) * math.sqrt(2 / size)
        basis[0] /= math.sqrt(2)
        self.register_buffer('basis', basis)
        self.register_buffer('frequency', n * math.pi / size)
        self.raw_speed = nn.Parameter(torch.tensor(math.log(math.expm1(init_speed))))
        self.direction = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.fixed_t, self.directional, self.size = fixed_t, directional, size

    def forward(self, x):
        if x.shape[-2:] != (self.size, self.size):
            raise ValueError(f'WPO expects {self.size}x{self.size}, got {tuple(x.shape[-2:])}')
        dtype = x.dtype
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        basis = self.basis.to(work.dtype)
        k = self.frequency.to(work.dtype)
        d = self.direction.tanh() if self.directional else torch.zeros_like(self.direction)
        radius2 = ((1 + d) * k[None, None, :, None].square() +
                   (1 - d) * k[None, None, None, :].square())
        radius = torch.sqrt(radius2.clamp_min(1e-12))
        response = torch.cos((F.softplus(self.raw_speed) + 1e-6) * self.fixed_t * radius)
        spectrum = basis @ work @ basis.T
        return (basis.T @ (spectrum * response) @ basis).to(dtype)


class BottleneckBottle(nn.Module):
    def __init__(self, config=BottleConfig()):
        super().__init__()
        self.config = config
        self.cpe = CPE(512)
        self.input_dw = RepDW(512, 3) if config.enable_input_repdw else nn.Identity()
        self.input_norm = nn.GroupNorm(8, 512)
        self.input_pw = nn.Conv2d(512, 256, 1)
        self.mamba_norm = nn.LayerNorm(64)
        self.mamba = Mamba(d_model=64, d_state=16, d_conv=4, expand=2)
        self.mamba_conv, self.mamba_gate = local(64), ChannelGate(64)
        self.gamma_m = nn.Parameter(torch.ones(1, 64, 1, 1))
        self.local3, self.local7 = local(96), local(96, 7)
        self.fuse = nn.Sequential(nn.Conv2d(256, 256, 1), nn.GroupNorm(8, 256), nn.GELU())
        self.query_proj, self.query_norm = nn.Conv2d(256, 128, 1), nn.LayerNorm(128)
        self.query_weight = nn.Parameter(torch.randn(128) / math.sqrt(128))
        self.query_mlp = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 256))
        nn.init.zeros_(self.query_mlp[-1].weight); nn.init.zeros_(self.query_mlp[-1].bias)
        self.delta_g = nn.Parameter(torch.tensor(1.0))
        self.wave_reduce, self.wave_norm = nn.Conv2d(256, 128, 1), nn.LayerNorm(128)
        self.wave = DirectionalWPO(fixed_t=config.fixed_t, init_speed=config.init_speed, directional=config.directional)
        self.wave_conv, self.wave_gate = local(128), ChannelGate(128)
        self.gamma_w = nn.Parameter(torch.ones(1, 128, 1, 1))
        self.wave_expand = nn.Conv2d(128, 256, 1, bias=False)  # zero update must project to exactly zero
        self.channel_mlp = nn.Sequential(nn.Conv2d(128, 256, 1), nn.GroupNorm(8, 256), nn.GELU(), nn.Conv2d(256, 128, 1))
        self.stage4_local, self.stage4_fuse = local(128), nn.Conv2d(256, 256, 1)
        self.gamma4 = nn.Parameter(torch.ones(1, 256, 1, 1))
        self.output_dw = RepDW(256, 3) if config.enable_output_repdw else nn.Identity()
        self.output_norm, self.output_pw = nn.GroupNorm(8, 256), nn.Conv2d(256, 512, 1)
        self.gamma_out = nn.Parameter(torch.full((1, 512, 1, 1), 1e-3))

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:] != (512, 8, 8):
            raise ValueError(f'Bottle expects [B,512,8,8], got {tuple(x.shape)}')
        x0 = self.cpe(x)
        t = self.input_pw(F.gelu(self.input_norm(self.input_dw(x0))))
        g, a, b = t.split((64, 96, 96), dim=1)
        gm = self.mamba_conv(g)
        if self.config.enable_mamba:
            tokens = g.flatten(2).transpose(1, 2)
            inc = self.mamba(self.mamba_norm(tokens)).transpose(1, 2).reshape_as(g)
            gm = gm + self.mamba_gate(g) * inc
        g = g + self.gamma_m * gm
        f = self.fuse(torch.cat((g, self.local3(a), self.local7(b)), dim=1))
        if self.config.enable_global_query:
            q = self.query_norm(self.query_proj(f).flatten(2).transpose(1, 2))
            weights = torch.softmax((q * self.query_weight).sum(-1) / math.sqrt(128), dim=1)
            summary = (weights.unsqueeze(-1) * q).sum(1)
            control = 1 + self.delta_g * (2 * torch.sigmoid(self.query_mlp(summary)) - 1)
            f = f * control[:, :, None, None]
        u = self.wave_reduce(f).permute(0, 2, 3, 1)
        u = self.wave_norm(u).permute(0, 3, 1, 2)
        update = self.wave_conv(u)
        if self.config.enable_dwpo:
            update = update + self.wave_gate(u) * self.wave(u)
        z = f + self.wave_expand(self.gamma_w * update)
        if self.config.enable_stage4:
            a, b = z.chunk(2, dim=1)
            z = z + self.gamma4 * self.stage4_fuse(torch.cat((self.channel_mlp(a), self.stage4_local(b)), dim=1))
        return x0 + self.gamma_out * self.output_pw(F.gelu(self.output_norm(self.output_dw(z))))

    def reparameterize_repdw(self):
        if self.training:
            raise RuntimeError('reparameterization requires eval mode')
        for module in list(self.modules()):
            if isinstance(module, RepDW):
                module.switch_to_deploy()


def build_candidate(pretrained=True, config=None):
    model = build_parent(pretrained=pretrained)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4607)
        model.decoder.e2 = BottleneckBottle(config or BottleConfig())
    return model

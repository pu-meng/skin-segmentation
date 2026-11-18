"""A1-S1 + Skip1-v0.7  (Haar-TinyViM + D-WPO, simplified gate, difference fusion).

Implements the user's v0.7 diagram exactly:

    S_hat = S + P(S)                       # SRP: split -> RepDW3 / RepDW7 -> 1x1
    D_hat = D + Q(D)                       # Q: 1x1
    F_H   = HaarTinyViM(S_hat)             # LL -> RepDW7 (large RF), [LH,HL,HH] -> RepDW3
    F_W   = D-WPO(S_hat)                   # 3x3 DW -> sigmoid(A_z + A_c) mask -> PW
    V     = PW([F_H, F_W])
    G_s   = sigmoid(Conv3x3([S_hat, D_hat]))     # simplified gate vs v0.6's three-source gate
    R     = PW(V * G_s)
    Y     = D_hat + S_hat + alpha * R            # alpha: learnable per-channel (64)

Two things to be aware of, both inherited from the diagram rather than chosen here:
1. Y carries D_hat + S_hat, so at init Y is ~ (S + D) instead of v0.6's ~S. alpha starts at
   0.01 like v0.6's gamma, so the fused term enters gently, but the mean level of the skip
   tensor is roughly doubled relative to v0.6.
2. The module still returns (Y, d) so the decoder's existing concat(skip_out, d) path and the
   D0 interface are untouched; only the S side is replaced.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from source.a1_s1_20261108.candidate import build_candidate as build_a1_s1


def conv1(ci, co, bias=True):
    return nn.Conv2d(ci, co, 1, bias=bias)


def dwt(x):
    """One-level Haar decomposition; each band is half resolution."""
    a = x[..., 0::2, 0::2]; b = x[..., 0::2, 1::2]
    c = x[..., 1::2, 0::2]; d = x[..., 1::2, 1::2]
    return ((a + b + c + d) * .5, (a - b + c - d) * .5,
            (a + b - c - d) * .5, (a - b - c + d) * .5)


def idwt(bands):
    ll, lh, hl, hh = bands
    a = (ll + lh + hl + hh) * .5; b = (ll - lh + hl - hh) * .5
    c = (ll + lh - hl - hh) * .5; d = (ll - lh - hl + hh) * .5
    rows = torch.stack((torch.stack((a, b), -1), torch.stack((c, d), -1)), -3)
    n, ch, h, _, w, _ = rows.shape
    return rows.reshape(n, ch, h * 2, w * 2)


class RepDW(nn.Module):
    """Reparameterisable depthwise block from the diagram (training form, branches summed).

    RepDW3: DW3x3 + DW1x1 + identity.   RepDW7: DW7x7 + DW1x7 + DW7x1 + identity.
    Each branch carries its own BatchNorm, exactly as drawn.
    """

    def __init__(self, c, k):
        super().__init__()
        assert k in (3, 7), k
        self.k = k
        branches = [nn.Sequential(nn.Conv2d(c, c, k, padding=k // 2, groups=c, bias=False),
                                  nn.BatchNorm2d(c))]
        if k == 3:
            branches.append(nn.Sequential(nn.Conv2d(c, c, 1, groups=c, bias=False),
                                          nn.BatchNorm2d(c)))
        else:
            branches.append(nn.Sequential(nn.Conv2d(c, c, (1, k), padding=(0, k // 2), groups=c, bias=False),
                                          nn.BatchNorm2d(c)))
            branches.append(nn.Sequential(nn.Conv2d(c, c, (k, 1), padding=(k // 2, 0), groups=c, bias=False),
                                          nn.BatchNorm2d(c)))
        branches.append(nn.Sequential(nn.Identity(), nn.BatchNorm2d(c)))
        self.branches = nn.ModuleList(branches)

    def forward(self, x):
        out = self.branches[0](x)
        for branch in self.branches[1:]:
            out = out + branch(x)
        return out


class SRP(nn.Module):
    """Split-Rep preprocessing: S_hat = S + P(S)."""

    def __init__(self, c=64):
        super().__init__()
        self.pre = conv1(c, c)
        half = c // 2
        self.b3 = RepDW(half, 3)
        self.b7 = RepDW(half, 7)
        self.post = conv1(c, c)

    def forward(self, s):
        x = self.pre(s)
        left, right = x.split(x.shape[1] // 2, dim=1)
        return s + self.post(torch.cat((self.b3(left), self.b7(right)), dim=1))


class HaarTinyViM(nn.Module):
    """LL -> RepDW7 (low frequency, large receptive field); LH/HL/HH -> RepDW3 (local detail).

    The three high-frequency bands keep their own channel groups through the 1x1 mix, so the
    depthwise stage never blends bands; idwt then reconstructs the full-resolution F_H.
    """

    def __init__(self, c=64):
        super().__init__()
        self.low = RepDW(c, 7)
        self.hf_reduce = conv1(3 * c, 3 * c)
        self.high = RepDW(3 * c, 3)
        self.out = conv1(c, c)

    def forward(self, s):
        ll, lh, hl, hh = dwt(s)
        f_low = self.low(ll)
        f_high = self.high(self.hf_reduce(torch.cat((lh, hl, hh), dim=1)))
        dl, dh, dv = f_high.split(f_high.shape[1] // 3, dim=1)
        return self.out(idwt((f_low, dl, dh, dv)))


class DirectionalWaveOperator(nn.Module):
    """A_z: one anisotropic frequency response per channel.

    lam = a_x * kx^2 + a_y * ky^2 with a_x = 1+tanh(dir), a_y = 1-tanh(dir), so `dir`
    controls how strongly the response favours the horizontal over the vertical axis
    (one bounded scalar per channel). The fractional order is exp(order) > 0.
    The response is zero-mean so it can only modulate, never shift the mean of a feature.
    """

    def __init__(self, c=64):
        super().__init__()
        self.dir = nn.Parameter(torch.zeros(c))
        self.order = nn.Parameter(torch.zeros(c))

    def forward(self, x):
        _, _, H, W = x.shape
        kx = torch.fft.fftfreq(W, device=x.device, dtype=torch.float32) * 2 * math.pi
        ky = torch.fft.fftfreq(H, device=x.device, dtype=torch.float32) * 2 * math.pi
        ax = 1.0 + torch.tanh(self.dir)
        ay = 1.0 - torch.tanh(self.dir)
        lam = ax[:, None, None] * kx[None, None, :].square() + ay[:, None, None] * ky[None, :, None].square()
        p = torch.exp(self.order)[:, None, None]
        resp = torch.exp(-(lam + 1e-8).pow(p))
        return resp - resp.mean((-2, -1), keepdim=True)


class DWPOBranch(nn.Module):
    """3x3 DW -> sigmoid(A_z + A_c) mask -> PW -> 1x1."""

    def __init__(self, c=64):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        self.wave = DirectionalWaveOperator(c)
        self.ch_gate = nn.Parameter(torch.zeros(c))
        self.pw = conv1(c, c)
        self.out = conv1(c, c)

    def forward(self, s):
        x = self.dw(s)
        mask = torch.sigmoid(self.wave(x) + self.ch_gate[None, :, None, None])
        return self.out(self.pw(x * mask))


class SkipV07(nn.Module):
    """Difference fusion: Y = D_hat + S_hat + alpha * R, returned as the new skip tensor."""

    def __init__(self, c=64):
        super().__init__()
        self.srp = SRP(c)
        self.q = conv1(c, c)
        self.haar = HaarTinyViM(c)
        self.dwpo = DWPOBranch(c)
        self.v = conv1(2 * c, c)
        self.gate = nn.Conv2d(2 * c, c, 3, padding=1)
        self.r = conv1(c, c)
        # LayerScale analogue of v0.6's gamma=0.01, now per channel.
        self.alpha = nn.Parameter(torch.full((1, c, 1, 1), 0.01))

    def forward(self, s, d):
        if s.shape != d.shape or s.ndim != 4 or s.shape[1:] != (64, 64, 64):
            raise ValueError(f'Skip v0.7 expects matching [B,64,64,64], got {tuple(s.shape)} and {tuple(d.shape)}')
        s_hat = self.srp(s)
        d_hat = d + self.q(d)
        f_h = self.haar(s_hat)
        f_w = self.dwpo(s_hat)
        v = self.v(torch.cat((f_h, f_w), dim=1))
        g = torch.sigmoid(self.gate(torch.cat((s_hat, d_hat), dim=1)))
        r = self.r(v * g)
        return d_hat + s_hat + self.alpha * r, d


def build_candidate(pretrained=True):
    model = build_a1_s1(pretrained=pretrained)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4707)
        model.decoder.e3 = SkipV07()
    return model

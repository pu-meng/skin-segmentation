"""A1 + Skip2/Skip3 dual-direction gated skip refinement (V1).

Only Skip2 (level1, 32x32/C128) and Skip3 (level2, 16x16/C256) are touched. Everything else
is the untouched A1 network: TinyViM-S encoder, bottleneck E2A, D3 DStage, D2 DualStage,
D1, Skip1 E3, D0 and the segmentation head all keep their original code.

Per-side pipeline (identical structure, independent parameters):

    X_hat = X + beta * F_mix(X)                    # content, PW -> split -> 3 branches -> PW
    F_mix = PWConv( cat([ Mamba(ch_m), RepDW3(ch_3), RepDW7(ch_7) ]) )

    R_cross = CrossGate_self(v_self=X_hat, v_other=Y_hat)   # own Q,K,V,O projections
    R_local = PWConv(DWConv3x3(X_hat))
    R_hat   = R_cross + rho * R_local                        # parallel, not serial

    X_out = X + alpha * R_hat                                # residual on the ORIGINAL X

Wiring (from the real A1 forward in source/e123_joint_20261105/candidate.py):

    x = stages[3](e2(f[3]))
    for level in (2,1,0):
        x = ups[level](x)            # level2 -> [B,256,16,16], level1 -> [B,128,32,32]
        s = f[level]
        if level == 2: s, x = skip3(s, x)
        if level == 1: s, x = skip2(s, x)
        if level == 0: s, x = e3(s, x)
        x = stages[level](cat([x, s], 1))   # cat order stays (decoder, encoder)

Deliberate choice recorded in the report: the diagram's "Q_d K_s^T / sqrt(C)" suggests a
(B,C,C) channel matrix, but "R = PW(V * G)" needs G to broadcast against V. This implements
a per-position channel dot product reduced to (B,1,H,W) — a spatial gate — which keeps the
"same position" reading and the elementwise product consistent. gate_mode='channel' switches
to the global variant for comparison.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from source.decoder_dual_a_20261106.candidate import build_candidate as build_a1


def conv1(ci, co, bias=True):
    return nn.Conv2d(ci, co, 1, bias=bias)


# ------------------------------------------------------------------ content branches

class RepDW(nn.Module):
    """Reparameterisable depthwise block used for the 3x3 / 7x7 content branches.

    RepDW3: DW3x3 + DW1x1 + identity (each with its own BN).
    RepDW7: DW7x7 + DW1x7 + DW7x1 + identity.
    All branches are summed in training form; switch_to_deploy() fuses them exactly.
    """

    def __init__(self, c, k):
        super().__init__()
        assert k in (3, 7), k
        self.c, self.k = c, k
        branches = [nn.Sequential(nn.Conv2d(c, c, k, padding=k // 2, groups=c, bias=False),
                                  nn.BatchNorm2d(c))]
        if k == 3:
            branches.append(nn.Sequential(nn.Conv2d(c, c, 1, groups=c, bias=False),
                                          nn.BatchNorm2d(c)))
        else:
            branches.append(nn.Sequential(nn.Conv2d(c, c, (1, k), padding=(0, k // 2),
                                                    groups=c, bias=False), nn.BatchNorm2d(c)))
            branches.append(nn.Sequential(nn.Conv2d(c, c, (k, 1), padding=(k // 2, 0),
                                                    groups=c, bias=False), nn.BatchNorm2d(c)))
        # No identity branch: ContentBlock applies `x + beta * mixed`, so an identity branch
        # here would duplicate the residual path. RepDW3 -> 2 branches, RepDW7 -> 3.
        self.branches = nn.ModuleList(branches)
        self.deploy = None

    @torch.no_grad()
    def switch_to_deploy(self):
        """Fuse every branch into one depthwise kernel. Identity keeps its BN scaling."""
        if self.deploy is not None:
            return
        k = self.k
        total_w = torch.zeros(self.c, 1, k, k)
        total_b = torch.zeros(self.c)
        for branch in self.branches:
            conv, bn = branch[0], branch[1]
            scale = bn.weight.detach() * torch.rsqrt(bn.running_var.detach() + bn.eps)
            bn_bias = bn.bias.detach() - bn.running_mean.detach() * scale
            if isinstance(conv, nn.Identity):
                w = torch.zeros(self.c, 1, k, k)
                w[:, 0, k // 2, k // 2] = 1.0
                total_w += w * scale[:, None, None, None]
                total_b += bn_bias
                continue
            w = conv.weight.detach() * scale[:, None, None, None]
            pad_h = (k - conv.kernel_size[0]) // 2
            pad_w = (k - conv.kernel_size[1]) // 2
            total_w += F.pad(w, (pad_w, pad_w, pad_h, pad_h))
            conv_bias = torch.zeros(self.c) if conv.bias is None else conv.bias.detach()
            total_b += conv_bias * scale + bn_bias
        fused = nn.Conv2d(self.c, self.c, k, padding=k // 2, groups=self.c, bias=True)
        fused.weight.copy_(total_w)
        fused.bias.copy_(total_b)
        fused.requires_grad_(False)
        self.deploy = fused
        self.branches = nn.ModuleList()      # release the training branches

    def forward(self, x):
        if self.deploy is not None:
            return self.deploy(x)
        out = self.branches[0](x)
        for branch in self.branches[1:]:
            out = out + branch(x)
        return out


class DWLargeBranch(nn.Module):
    """Large-kernel depthwise branch: local context with a big receptive field, no SSM cost.

    Used at Skip2 (32x32 / 1024 tokens) where a Mamba scan was the single most expensive
    operation in the module. 11x11 depthwise keeps a 11-pixel receptive radius at 32x32,
    which covers the same neighbourhood as the 7x7 RepDW but without channel splitting.
    """

    def __init__(self, c, k=11):
        super().__init__()
        self.dw = nn.Conv2d(c, c, k, padding=k // 2, groups=c, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.dw(x)))


class MambaBranch(nn.Module):
    """Unchanged official Mamba-1 internals; `impl='dw'` is a CPU-only stand-in for tests."""

    def __init__(self, c, impl='mamba'):
        super().__init__()
        self.impl = impl
        if impl == 'mamba':
            from mamba_ssm.modules.mamba_simple import Mamba
            self.norm = nn.LayerNorm(c)
            self.mixer = Mamba(d_model=c, d_state=16, d_conv=4, expand=2)
        else:
            self.net = nn.Sequential(nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False),
                                     nn.BatchNorm2d(c), nn.GELU())

    def forward(self, x):
        if self.impl != 'mamba':
            return self.net(x)
        b, c, h, w = x.shape
        t = x.flatten(2).transpose(1, 2)
        return self.mixer(self.norm(t)).transpose(1, 2).reshape(b, c, h, w)


class ContentBlock(nn.Module):
    """X_hat = X + beta * PW_{4C->C}(GELU(PW_{C->4C}(BN(DW7(X)))))

    ConvNeXt-style inverted bottleneck. Replaces three narrow parallel branches (Mamba/DW11 +
    RepDW3 + RepDW7, 6-8 kernels per side) with one large-kernel sequence (5 kernels). The 4C
    width and 7x7 depthwise recover the receptive field and capacity that the dropped branches
    provided, while the operator count -- which the profiler showed is the real cost driver --
    drops by a third.
    """

    def __init__(self, c, expand=4, beta=1.0):
        super().__init__()
        self.expand = expand
        hidden = c * expand
        self.dw = nn.Conv2d(c, c, 7, padding=3, groups=c, bias=False)
        self.norm = nn.BatchNorm2d(c)
        self.pw_in = nn.Conv2d(c, hidden, 1, bias=False)
        self.act = nn.GELU()
        self.pw_out = nn.Conv2d(hidden, c, 1, bias=False)
        self.beta = nn.Parameter(torch.full((1, c, 1, 1), float(beta)))

    def forward(self, x):
        z = self.norm(self.dw(x))
        z = self.pw_out(self.act(self.pw_in(z)))
        return x + self.beta * z


# ------------------------------------------------------------------ gating + local

class CrossGate(nn.Module):
    """One direction only: own Q, K, V and output projection (no sharing across directions).

    Scale-normalised gate. The raw dot product (q*k).sum/sqrt(C) grows with both the channel
    count and the feature magnitude, which made Skip3's correction explode to 90% of the input
    within 10 epochs. Using the cosine similarity keeps the gate in [-1,1] regardless of C or
    scale; `tau` (learnable, init 1.0) sets how sharp the sigmoid is.

        G = sigmoid(tau * cos(q, k))       per position, over channels
        R_cross = PW_o(V * G)
    """

    def __init__(self, c, mode='spatial', ratio=4):
        super().__init__()
        self.mode = mode
        # Four full C->C 1x1 projections cost 4*C^2*HW and dominate the module on small maps.
        # Projecting into C/4 keeps the gate semantics and cuts both MACs and params 4x.
        d = max(8, c // ratio)
        self.wq = conv1(c, d, bias=False)
        self.wk = conv1(c, d, bias=False)
        self.wv = conv1(c, d, bias=False)
        self.wo = conv1(d, c, bias=False)
        self.tau = nn.Parameter(torch.ones(1))

    def forward(self, v_self, v_other):
        q = self.wq(v_self)
        k = self.wk(v_other)
        if self.mode == 'spatial':
            qn = F.normalize(q, dim=1, eps=1e-6)
            kn = F.normalize(k, dim=1, eps=1e-6)
            cos = (qn * kn).sum(1, keepdim=True)
        else:
            qn = F.normalize(q.mean((2, 3), keepdim=True), dim=1, eps=1e-6)
            kn = F.normalize(k.mean((2, 3), keepdim=True), dim=1, eps=1e-6)
            cos = (qn * kn).sum(1, keepdim=True)
        gate = torch.sigmoid(self.tau * cos)
        return self.wo(self.wv(v_self) * gate)


class LocalConv(nn.Module):
    """R_local = DW5x5(X) — the parallel convolution protection branch.

    Dropped the trailing pointwise: the cross branch already mixes channels, so this branch only
    needs to supply a local, channel-wise correction. One kernel instead of two.
    """

    def __init__(self, c, k=5):
        super().__init__()
        self.dw = nn.Conv2d(c, c, k, padding=k // 2, groups=c, bias=False)

    def forward(self, x):
        return self.dw(x)


# ------------------------------------------------------------------ the skip module

class SkipS2S3(nn.Module):
    """Dual-direction gated skip for Skip2 (C=128) / Skip3 (C=256)."""

    def __init__(self, c, expand=4, impl='mamba',
                 use_gate=True, use_local=True, gate_mode='spatial', beta=1.0, rho=0.1):
        super().__init__()
        self.c = c
        self.use_gate = use_gate
        self.use_local = use_local
        self.content_s = ContentBlock(c, expand, beta)
        self.content_d = ContentBlock(c, expand, beta)
        if use_gate:
            self.gate_s = CrossGate(c, gate_mode)
            self.gate_d = CrossGate(c, gate_mode)
        if use_local:
            self.local_s = LocalConv(c)
            self.local_d = LocalConv(c)
            # Confirmed: rho = 0.1 per channel, learnable. Cross-gating is the core path,
            # local convolution is auxiliary; both are summed in parallel.
            self.rho_s = nn.Parameter(torch.full((1, c, 1, 1), float(rho)))
            self.rho_d = nn.Parameter(torch.full((1, c, 1, 1), float(rho)))
        # Controlled residual on the ORIGINAL features, per channel, init 0.01.
        self.alpha_s = nn.Parameter(torch.full((1, c, 1, 1), 0.01))
        self.alpha_d = nn.Parameter(torch.full((1, c, 1, 1), 0.01))
        # Evaluation-time switches (plain dict: never serialised, adds no parameters).
        self.runtime = {'enabled': True, 'use_gate': True, 'use_local': True}
        self.collect_stats = False
        self.stats = None

    def forward(self, s, d):
        if not self.runtime.get('enabled', True):
            return s, d

        s_hat = self.content_s(s)
        d_hat = self.content_d(d)

        # Two parallel corrections on the same content features; with every switch off the
        # correction is exactly zero, so the module degenerates to identity regardless of alpha.
        cross_s = cross_d = None
        local_s = local_d = None
        r_hat_s = torch.zeros_like(s_hat)
        r_hat_d = torch.zeros_like(d_hat)
        if self.use_gate and self.runtime.get('use_gate', True):
            cross_s = self.gate_s(s_hat, d_hat)
            cross_d = self.gate_d(d_hat, s_hat)
            r_hat_s = r_hat_s + cross_s
            r_hat_d = r_hat_d + cross_d
        if self.use_local and self.runtime.get('use_local', True):
            local_s = self.rho_s * self.local_s(s_hat)
            local_d = self.rho_d * self.local_d(d_hat)
            r_hat_s = r_hat_s + local_s
            r_hat_d = r_hat_d + local_d

        if self.collect_stats:
            with torch.no_grad():
                eps = 1e-8
                self.stats = {
                    'alpha_s': float(self.alpha_s.mean()),
                    'alpha_d': float(self.alpha_d.mean()),
                    'beta': float(self.content_s.beta.mean()),
                    'rho': float(self.rho_s.mean()),
                    's_norm': float(s.norm()),
                    'd_norm': float(d.norm()),
                    's_hat_over_s': float(s_hat.norm() / (s.norm() + eps)),
                    'corr_s_over_s': float((self.alpha_s * r_hat_s).norm() / (s.norm() + eps)),
                    'corr_d_over_d': float((self.alpha_d * r_hat_d).norm() / (d.norm() + eps)),
                    'tau': float(self.gate_s.tau) if self.use_gate else 0.0,
                    'cross_over_s_hat': float(cross_s.norm() / (s_hat.norm() + eps)) if cross_s is not None else 0.0,
                    'local_over_s_hat': float(local_s.norm() / (s_hat.norm() + eps)) if local_s is not None else 0.0,
                    'cross_share': (float(cross_s.norm() / (cross_s.norm() + local_s.norm() + eps))
                                    if (cross_s is not None and local_s is not None) else 0.0),
                }

        return s + self.alpha_s * r_hat_s, d + self.alpha_d * r_hat_d


# ------------------------------------------------------------------ decoder wiring

class DecoderS2S3(nn.Module):
    """Same decoder as A1, with Skip3 at level2 and Skip2 at level1 inserted before concat."""

    def __init__(self, old, skip2=None, skip3=None):
        super().__init__()
        self.stages = old.stages
        self.ups = old.ups
        self.head = old.head
        self.e2 = old.e2
        self.e3 = old.e3
        self.skip2 = skip2
        self.skip3 = skip3

    def forward(self, f):
        x = self.stages[3](self.e2(f[3]))
        for level in (2, 1, 0):
            x = self.ups[level](x)
            s = f[level]
            if level == 2 and self.skip3 is not None:
                s, x = self.skip3(s, x)
            if level == 1 and self.skip2 is not None:
                s, x = self.skip2(s, x)
            if level == 0:
                s, x = self.e3(s, x)
            x = self.stages[level](torch.cat([x, s], 1))
        return torch.sigmoid(F.interpolate(self.head(x), size=(256, 256),
                                           mode='bilinear', align_corners=False))


SKIP_CHANNELS = {
    # name: (C, expand) -- one 7x7 depthwise + 4C bottleneck per side.
    'skip2': (128, 4),
    'skip3': (256, 4),
}


def build_candidate(pretrained=True, use_skip2=True, use_skip3=True,
                    use_gate=True, use_local=True, gate_mode='spatial',
                    impl='mamba', beta=1.0, rho=0.1):
    model = build_a1(pretrained=pretrained)
    old = model.decoder
    skip2 = skip3 = None
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4202)
        if use_skip3:
            c, expand = SKIP_CHANNELS['skip3']
            skip3 = SkipS2S3(c, expand, impl, use_gate, use_local, gate_mode, beta, rho)
        torch.manual_seed(4203)
        if use_skip2:
            c, expand = SKIP_CHANNELS['skip2']
            skip2 = SkipS2S3(c, expand, impl, use_gate, use_local, gate_mode, beta, rho)
    model.decoder = DecoderS2S3(old, skip2, skip3)
    return model

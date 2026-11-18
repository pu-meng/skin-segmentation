"""Independent V2 candidate: channel-shared A and fixed mean state readout.

Keeps expand=2, rho->delta, B injection without delta, D skip, SiLU(Z),
input/output projections, outer LN/residual. No learned C generator remains.
"""
import torch
from torch import nn
from torch.nn import functional as F
from Models.TinyViM.mamba_bottleneck_v2 import NoDeltaInjectionMamba
from Models.TinyViM.mamba_bottleneck_v2_triton_pilot import scan_v2_triton


def shared_a_mean_scan(u, delta, A, B):
    if A.shape != (1, B.shape[1]):
        raise ValueError('A must be [1,N], B must be [batch,N,length]')
    # expand's backward sums all channel contributions into the single A row.
    # Fixed C=1/N implements mean(h) with the existing audited V2 scan kernel.
    # C is a runtime constant, not a parameter or an input-generated coefficient.
    fixed_readout = torch.full_like(B, 1.0 / B.shape[1], requires_grad=False)
    return scan_v2_triton(u, delta, A.expand(u.shape[1], -1), B, fixed_readout)


class SharedAMeanMambaV2(NoDeltaInjectionMamba):
    def __init__(self, d_model, d_state=16, d_conv=4):
        super().__init__(d_model, d_state=d_state, d_conv=d_conv, expand=2)
        # Retain exactly the parent's rho and B initialization, remove C rows.
        with torch.random.fork_rng(devices=[]):
            reduced = nn.Linear(self.d_inner, self.dt_rank + self.d_state, bias=False)
        with torch.no_grad():
            reduced.weight.copy_(self.x_proj.weight[:self.dt_rank + self.d_state])
        self.x_proj = reduced
        self.A_log = nn.Parameter(self.A_log.detach()[:1].clone())
        self.A_log._no_weight_decay = True

    def forward(self, hidden_states, inference_params=None):
        if inference_params is not None:
            raise NotImplementedError('Full image sequences only')
        f, z = self.in_proj(hidden_states).transpose(1, 2).chunk(2, dim=1)
        t = self.act(self.conv1d(f)[..., :hidden_states.shape[1]])
        rho, B = torch.split(self.x_proj(t.transpose(1, 2)),
                             [self.dt_rank, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(rho)).transpose(1, 2)
        A = -torch.exp(self.A_log.float())
        y = shared_a_mean_scan(t, delta, A, B.transpose(1, 2))
        y = (y + self.D[None, :, None] * t) * F.silu(z)
        return self.out_proj(y.transpose(1, 2))


class BottleneckMambaV2SharedAMean(nn.Module):
    def __init__(self, channels=512, d_state=16, seed=44):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.norm = nn.LayerNorm(channels)
            self.mixer = SharedAMeanMambaV2(channels, d_state=d_state)

    def forward(self, feature):
        b, c, h, w = feature.shape
        tokens = feature.flatten(2).transpose(1, 2)
        tokens = tokens + self.mixer(self.norm(tokens))
        return tokens.transpose(1, 2).reshape(b, c, h, w)

"""User V2 only: canonical Mamba-1 generator, but J=B*U (no delta).

No package files or active V1 sources are modified. Explicit differentiable
recurrence changes only the injection coefficient; it does not divide by delta
or silently alter the D skip and z gate.
"""
import torch
from torch import nn
from torch.nn import functional as F
from mamba_ssm.modules.mamba_simple import Mamba


def scan_without_delta_injection(u, delta, A, B, C):
    """u/delta [b,E,L], A [E,N], B/C [b,N,L] -> output [b,E,L]."""
    original_dtype = u.dtype
    u, delta, A, B, C = [t.float() for t in (u, delta, A, B, C)]
    # R [b,E,L,N] is unchanged. J [b,E,L,N] deliberately excludes delta.
    transitions = torch.exp(torch.einsum('bel,en->beln', delta, A))
    injections = torch.einsum('bel,bnl->beln', u, B)
    state = torch.zeros(u.shape[0], u.shape[1], A.shape[1], device=u.device, dtype=u.dtype)
    outputs = []
    for t in range(u.shape[-1]):
        state = transitions[:, :, t] * state + injections[:, :, t]
        outputs.append((state * C[:, None, :, t]).sum(-1))
    return torch.stack(outputs, dim=-1).to(original_dtype)


class NoDeltaInjectionMamba(Mamba):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, **kwargs):
        kwargs['use_fast_path'] = False
        super().__init__(d_model, d_state=d_state, d_conv=d_conv, expand=expand, **kwargs)

    def forward(self, hidden_states, inference_params=None):
        if inference_params is not None:
            raise NotImplementedError('Full image sequences only; cached decoding not implemented')
        x, z = self.in_proj(hidden_states).transpose(1, 2).chunk(2, dim=1)
        u = self.act(self.conv1d(x)[..., :hidden_states.shape[1]])
        # Original x_proj and rho->delta projection remain, unlike V1.
        q = self.x_proj(u.transpose(1, 2))
        rho, B, C = torch.split(q, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(rho)).transpose(1, 2)
        A = -torch.exp(self.A_log.float())
        y = scan_without_delta_injection(u, delta, A, B.transpose(1, 2), C.transpose(1, 2))
        y = (y + self.D[None, :, None] * u) * F.silu(z)
        return self.out_proj(y.transpose(1, 2))

    def step(self, *args, **kwargs):
        raise NotImplementedError('Cached decoding is outside this segmentation experiment')


class BottleneckMambaV2(nn.Module):
    def __init__(self, channels=512, d_state=16, seed=44):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.norm = nn.LayerNorm(channels)
            self.mixer = NoDeltaInjectionMamba(channels, d_state=d_state, expand=2)

    def forward(self, feature):
        b, c, h, w = feature.shape
        tokens = feature.flatten(2).transpose(1, 2)
        tokens = tokens + self.mixer(self.norm(tokens))
        return tokens.transpose(1, 2).reshape(b, c, h, w)

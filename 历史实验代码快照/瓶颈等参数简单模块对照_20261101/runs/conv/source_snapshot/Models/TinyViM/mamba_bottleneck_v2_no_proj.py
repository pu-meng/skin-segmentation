"""V2 without boundary projections: D=C0, F=Z=LN(X), full B/C recurrence.

Retains the original dt rank, initialization, causal convolution, trainable
per-channel A_log/D, no-delta injection, C readout, gate and outer residual.
CUDA uses the existing direct V2 Triton scan; CPU uses its reference fallback.
"""
import torch
from torch import nn
from torch.nn import functional as F
from Models.TinyViM.mamba_bottleneck_v2 import NoDeltaInjectionMamba
from Models.TinyViM.mamba_bottleneck_v2_triton_pilot import scan_v2_triton


class NoProjectionMambaV2(NoDeltaInjectionMamba):
    def __init__(self, d_model, d_state=16, d_conv=4):
        # Preserve Mamba's initialization recipe at the requested width.
        super().__init__(d_model, d_state=d_state, d_conv=d_conv, expand=1)
        del self.in_proj
        del self.out_proj

    def forward(self, hidden_states, inference_params=None):
        if inference_params is not None:
            raise NotImplementedError('Full image sequences only')
        f = z = hidden_states.transpose(1, 2)
        t = self.act(self.conv1d(f)[..., :hidden_states.shape[1]])
        rho, B, C = torch.split(self.x_proj(t.transpose(1, 2)),
                                [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(rho)).transpose(1, 2)
        A = -torch.exp(self.A_log.float())
        y = scan_v2_triton(t, delta, A, B.transpose(1, 2), C.transpose(1, 2))
        return ((y + self.D[None, :, None] * t) * F.silu(z)).transpose(1, 2)


class BottleneckMambaV2NoProjection(nn.Module):
    def __init__(self, channels=512, d_state=16, seed=44):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.norm = nn.LayerNorm(channels)
            self.mixer = NoProjectionMambaV2(channels, d_state=d_state)

    def forward(self, feature):
        b, c, h, w = feature.shape
        tokens = feature.flatten(2).transpose(1, 2)
        tokens = tokens + self.mixer(self.norm(tokens))
        return tokens.transpose(1, 2).reshape(b, c, h, w)

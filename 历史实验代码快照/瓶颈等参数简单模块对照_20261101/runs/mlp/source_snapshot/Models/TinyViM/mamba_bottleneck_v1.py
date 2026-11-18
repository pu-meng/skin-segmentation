"""User V1: direct nonlinear (delta, B, C) generator, unchanged Mamba-1 scan.

Placement is implemented by the separate experiment runner, not by editing the
shared TinyViM model or any installed Mamba package. CPU uses the official
reference scan for tests; CUDA uses the installed selective scan/autograd.
"""
import torch
from torch import nn
from torch.nn import functional as F
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref


class CausalParameterGenerator(nn.Module):
    """[b,E,L] -> DWConv3 -> SiLU -> pointwise -> [b,E+2N,L]."""
    def __init__(self, channels, d_state, delta_bias):
        super().__init__()
        self.depthwise = nn.Conv1d(channels, channels, 3, padding=2,
                                   groups=channels, bias=True)
        self.activation = nn.SiLU()
        self.pointwise = nn.Conv1d(channels, channels + 2*d_state, 1, bias=True)
        with torch.no_grad():
            self.pointwise.bias.zero_()
            # Preserve the standard initial delta timescale, without rho/dt_proj.
            self.pointwise.bias[:channels].copy_(delta_bias)

    def forward(self, u):
        length = u.shape[-1]
        return self.pointwise(self.activation(self.depthwise(u)[..., :length]))


class DirectParameterMamba(Mamba):
    """Version 1 only: remove x_proj and dt_proj, retain Delta*B*U injection."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, **kwargs):
        kwargs['use_fast_path'] = False
        super().__init__(d_model, d_state=d_state, d_conv=d_conv, expand=expand, **kwargs)
        with torch.random.fork_rng(devices=[]):
            self.parameter_generator = CausalParameterGenerator(
                self.d_inner, self.d_state, self.dt_proj.bias.detach())
        del self.x_proj
        del self.dt_proj

    def generate_parameters(self, u):
        # u [b,E,L] -> Q [b,E+2N,L] -> raw_delta [b,E,L], B/C [b,N,L].
        q = self.parameter_generator(u)
        return tuple(t.contiguous() for t in torch.split(
            q, [self.d_inner, self.d_state, self.d_state], dim=1))

    def forward(self, hidden_states, inference_params=None):
        if inference_params is not None:
            raise NotImplementedError('This experiment supports full-sequence training, not cached decoding')
        # [b,L,D] -> [b,2E,L] -> x/z each [b,E,L].
        x, z = self.in_proj(hidden_states).transpose(1, 2).chunk(2, dim=1)
        u = self.act(self.conv1d(x)[..., :hidden_states.shape[1]])
        raw_delta, B, C = self.generate_parameters(u)
        A = -torch.exp(self.A_log.float())
        scan = selective_scan_fn if u.is_cuda else selective_scan_ref
        # The original scan computes softplus(delta), exp(delta*A), and delta*B*u.
        # No delta_bias: the first E generator outputs already contain its bias.
        y = scan(u.contiguous(), raw_delta, A, B, C, self.D.float(),
                 z=z.contiguous(), delta_bias=None, delta_softplus=True)
        return self.out_proj(y.transpose(1, 2))

    def step(self, *args, **kwargs):
        raise NotImplementedError('Cached decoding is outside this image segmentation experiment')


class BottleneckMambaV1(nn.Module):
    """[b,512,8,8] -> LN + row-major Mamba-1 -> residual [b,512,8,8]."""
    def __init__(self, channels=512, d_state=16, seed=44):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.norm = nn.LayerNorm(channels)
            self.mixer = DirectParameterMamba(channels, d_state=d_state, expand=2)

    def forward(self, feature):
        b, c, h, w = feature.shape
        tokens = feature.flatten(2).transpose(1, 2)  # [b,c,h,w] -> [b,h*w,c]
        tokens = tokens + self.mixer(self.norm(tokens))
        return tokens.transpose(1, 2).reshape(b, c, h, w)

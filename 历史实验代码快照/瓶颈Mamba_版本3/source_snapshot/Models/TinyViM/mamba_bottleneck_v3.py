"""V3: spatial convolution plus input-dependent gates, without recurrent scan.

Y1 = GU*U + D*delta*T; GU=sum_n exp(delta*A_n). No B/C or extra D*T.
The inherited Mamba initialization supplies in/out, causal DWConv4, dt, A, D.
"""
import torch
from torch import nn
from torch.nn import functional as F
from mamba_ssm.modules.mamba_simple import Mamba


class ConvGateMamba(Mamba):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, conv_expand=4):
        super().__init__(d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                         use_fast_path=False)
        # Keep exactly the original rho rows and remove all B/C parameters.
        rho_proj = nn.Linear(self.d_inner, self.dt_rank, bias=False)
        with torch.no_grad():
            rho_proj.weight.copy_(self.x_proj.weight[:self.dt_rank])
        self.x_proj = rho_proj
        self.conv_expand = conv_expand
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(self.d_inner, self.d_inner, 3, padding=1,
                      groups=self.d_inner, bias=True),
            nn.GroupNorm(self.d_inner, self.d_inner),
            nn.Conv2d(self.d_inner, conv_expand*self.d_inner, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(conv_expand*self.d_inner, self.d_inner, 1, bias=True),
        )

    def forward(self, hidden_states, spatial_shape, inference_params=None):
        if inference_params is not None:
            raise NotImplementedError('Full spatial features only')
        b, length, _ = hidden_states.shape
        h, w = spatial_shape
        if h*w != length:
            raise ValueError('Spatial dimensions must match row-major token length')
        f, z = self.in_proj(hidden_states).transpose(1, 2).chunk(2, dim=1)
        t = self.act(self.conv1d(f)[..., :length])
        delta = F.softplus(self.dt_proj(self.x_proj(t.transpose(1, 2)))).transpose(1, 2)
        a = -torch.exp(self.A_log.float())
        gu = torch.exp(delta.float().unsqueeze(-1)*a[None, :, None, :]).sum(-1)
        u = self.spatial_conv(t.reshape(b, self.d_inner, h, w)).flatten(2)
        y1 = gu*u + self.D[None, :, None]*delta*t
        y2 = y1*F.silu(z)
        return self.out_proj(y2.transpose(1, 2))

    def step(self, *args, **kwargs):
        raise NotImplementedError('V3 uses spatial convolution, no recurrent state')


class BottleneckMambaV3(nn.Module):
    def __init__(self, channels=512, d_state=16, seed=44, conv_expand=4):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.norm = nn.LayerNorm(channels)
            self.mixer = ConvGateMamba(channels, d_state=d_state, conv_expand=conv_expand)

    def forward(self, feature):
        b, c, h, w = feature.shape
        tokens = feature.flatten(2).transpose(1, 2)
        tokens = tokens + self.mixer(self.norm(tokens), (h, w))
        return tokens.transpose(1, 2).reshape(b, c, h, w)

"""V2 with spatial depthwise-separable input/output projections.

Preserves expand=2, original projection channel weights and all V2 internals.
Each boundary projection is DWConv2d3 (identity initialized) -> PWConv2d1.
No extra activation, normalization or bias. Spatial context is noncausal;
the inner sequence convolution and V2 state recurrence remain unchanged.
"""
import torch
from torch import nn
from torch.nn import functional as F
from Models.TinyViM.mamba_bottleneck_v2_triton_pilot import DirectTritonMamba, scan_v2_triton


def separable_projection(linear):
    channels = linear.in_features
    # The added layers do not advance the caller's RNG stream.
    with torch.random.fork_rng(devices=[]):
        depthwise = nn.Conv2d(channels, channels, 3, padding=1,
                              groups=channels, bias=False)
        pointwise = nn.Conv2d(channels, linear.out_features, 1, bias=False)
    with torch.no_grad():
        depthwise.weight.zero_()
        depthwise.weight[:, 0, 1, 1] = 1
        pointwise.weight.copy_(linear.weight[:, :, None, None])
    return nn.Sequential(depthwise, pointwise)


class SpatialProjectionMambaV2(DirectTritonMamba):
    def __init__(self, d_model, d_state=16, d_conv=4):
        super().__init__(d_model, d_state=d_state, d_conv=d_conv, expand=2)
        self.in_proj = separable_projection(self.in_proj)
        self.out_proj = separable_projection(self.out_proj)

    def forward(self, hidden_states, spatial_shape, inference_params=None):
        if inference_params is not None:
            raise NotImplementedError('Full image sequences only')
        batch, length, channels = hidden_states.shape
        height, width = spatial_shape
        if height * width != length or channels != self.d_model:
            raise ValueError('Sequence and spatial shape differ')
        feature = hidden_states.transpose(1, 2).reshape(batch, channels, height, width)
        f, z = self.in_proj(feature).flatten(2).chunk(2, dim=1)
        t = self.act(self.conv1d(f)[..., :length])
        rho, B, C = torch.split(self.x_proj(t.transpose(1, 2)),
                                [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(rho)).transpose(1, 2)
        A = -torch.exp(self.A_log.float())
        y = scan_v2_triton(t, delta, A, B.transpose(1, 2), C.transpose(1, 2))
        y = (y + self.D[None, :, None] * t) * F.silu(z)
        y = y.reshape(batch, self.d_inner, height, width)
        return self.out_proj(y).flatten(2).transpose(1, 2)


class BottleneckMambaV2SeparableProjection(nn.Module):
    def __init__(self, channels=512, d_state=16, seed=44):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.norm = nn.LayerNorm(channels)
            self.mixer = SpatialProjectionMambaV2(channels, d_state=d_state)

    def forward(self, feature):
        b, c, h, w = feature.shape
        tokens = feature.flatten(2).transpose(1, 2)
        tokens = tokens + self.mixer(self.norm(tokens), (h, w))
        return tokens.transpose(1, 2).reshape(b, c, h, w)

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import MultiScaleSpatialFeatureBlock
from timm.layers import trunc_normal_


def rotate_every_two(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack([-x2, x1], dim=-1)
    return x.flatten(-2)


def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)


class RoPE2D(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        angle = 1.0 / (10000 ** torch.linspace(0, 1, self.head_dim // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.register_buffer('angle', angle)

    def forward(self, H: int, W: int, device=None):
        index = torch.arange(H * W, device=device or self.angle.device)

        # 计算sin和cos
        sin = torch.sin(index[:, None] * self.angle[None, :])  # (H*W, head_dim)
        cos = torch.cos(index[:, None] * self.angle[None, :])  # (H*W, head_dim)

        # 重塑为2D格式
        sin = sin.reshape(H, W, -1)  # (H, W, head_dim)
        cos = cos.reshape(H, W, -1)  # (H, W, head_dim)

        return sin, cos


class FourierAttention(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, value_head_dim: int,
                 freq_modulation_weight=None, freq_modulation_bias=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.value_head_dim = value_head_dim
        self.scaling = head_dim ** -0.5

        if freq_modulation_weight is None:
            self.freq_modulation_weight = nn.Parameter(torch.ones(num_heads, 1, 1))
        else:
            self.freq_modulation_weight = freq_modulation_weight

        if freq_modulation_bias is None:
            self.freq_modulation_bias = nn.Parameter(torch.zeros(num_heads, 1, 1))
        else:
            self.freq_modulation_bias = freq_modulation_bias

    def complex_activation(self, z):
        mag = torch.abs(z)
        mag_act = F.gelu(mag)
        scale = mag_act / (mag + 1e-6)
        return z * scale

    def forward(self, query_heads, key_heads, value_heads, freq_weight, H, W, sin=None, cos=None):
        B = query_heads.shape[0]

        if sin is not None and cos is not None:
            sin_flat = sin.view(H*W, -1)  # (HW, head_dim)
            cos_flat = cos.view(H*W, -1)  # (HW, head_dim)

            query_heads = theta_shift(query_heads, sin_flat.unsqueeze(0).unsqueeze(0), cos_flat.unsqueeze(0).unsqueeze(0))
            key_heads = theta_shift(key_heads, sin_flat.unsqueeze(0).unsqueeze(0), cos_flat.unsqueeze(0).unsqueeze(0))

        query_2d = query_heads.reshape(B, self.num_heads, H, W, self.head_dim)
        key_2d = key_heads.reshape(B, self.num_heads, H, W, self.head_dim)
        value_2d = value_heads.reshape(B, self.num_heads, H, W, self.value_head_dim)

        # 2D FFT
        query_fft = torch.fft.rfft2(query_2d, dim=(2, 3), norm='ortho')
        key_fft = torch.fft.rfft2(key_2d, dim=(2, 3), norm='ortho')
        value_fft = torch.fft.rfft2(value_2d, dim=(2, 3), norm='ortho')

        freq_mod_weight = self.freq_modulation_weight.unsqueeze(0).unsqueeze(-1) * freq_weight.view(B, self.num_heads, 1, 1, 1)
        freq_mod_bias = self.freq_modulation_bias.unsqueeze(0).unsqueeze(-1) * freq_weight.view(B, self.num_heads, 1, 1, 1)

        key_fft_mod = key_fft * freq_mod_weight + freq_mod_bias
        value_fft_mod = value_fft * freq_mod_weight + freq_mod_bias

        key_fft_act = self.complex_activation(key_fft_mod)
        value_fft_act = self.complex_activation(value_fft_mod)
        qk_fft = query_fft * torch.conj(key_fft_act)
        qk_freq_spatial = torch.fft.irfft2(qk_fft, s=(H, W), dim=(2, 3), norm='ortho')
        qk_freq_spatial = qk_freq_spatial.sum(dim=-1)  # (B, num_heads, H, W)
        qk_freq_spatial = qk_freq_spatial.reshape(B, self.num_heads, H*W)

        qk_freq_spatial = qk_freq_spatial * self.scaling

        attn_freq = F.softmax(qk_freq_spatial.unsqueeze(-1), dim=-2)

        value_freq_out = torch.fft.irfft2(value_fft_act, s=(H, W), dim=(2, 3), norm='ortho')
        value_freq_out = value_freq_out.reshape(B, self.num_heads, H*W, self.value_head_dim)
        out_freq = value_freq_out * attn_freq

        return out_freq


class SpectralSpatialAttention(nn.Module):
    def __init__(self, in_channels: int = 512, out_channels: int = 512,
                 ratio: int = 2, num_heads: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads

        self.key_channels = in_channels // ratio
        self.value_channels = in_channels // ratio
        self.head_dim = self.key_channels // num_heads
        self.value_head_dim = self.value_channels // num_heads
        self.scaling = self.head_dim ** -0.5

        self.InitialConvQ = nn.Conv2d(in_channels, self.key_channels, 1, bias=True)
        self.InitialConvK = nn.Conv2d(in_channels, self.key_channels, 1, bias=True)
        self.InitialConvV = nn.Conv2d(in_channels, self.value_channels, 1, bias=True)

        self.SpatialFeatureK = MultiScaleSpatialFeatureBlock(self.key_channels)
        self.SpatialFeatureV = MultiScaleSpatialFeatureBlock(self.value_channels)

        self.fourier_attention = FourierAttention(
            num_heads=num_heads,
            head_dim=self.head_dim,
            value_head_dim=self.value_head_dim
        )

        self.rope_2d = RoPE2D(self.key_channels, num_heads)

        self.context_proj = nn.Sequential(
            nn.Linear(self.key_channels, self.key_channels),
            nn.GELU(),
            nn.Linear(self.key_channels, num_heads)
        )

        self.pool_weight = nn.Parameter(torch.tensor(0.1))
        self.freq_weight = nn.Parameter(torch.tensor(0.9))

        self.ReorganizeConv = nn.Conv2d(self.value_channels, out_channels, 1, bias=True)
        self.norm = nn.LayerNorm(self.value_channels)
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device

        query = self.InitialConvQ(x)  # (B, key_channels, H, W)
        key = self.InitialConvK(x)
        value = self.InitialConvV(x)

        key_features = self.SpatialFeatureK(key)  # (B, L_extended, key_channels)
        value_features = self.SpatialFeatureV(value)  # (B, L_extended, value_channels)

        L_spatial = H * W
        key_spatial = key_features[:, :L_spatial, :]  # (B, HW, key_channels)
        value_spatial = value_features[:, :L_spatial, :]  # (B, HW, value_channels)

        query_flat = query.view(B, self.key_channels, -1).permute(0, 2, 1)  # (B, HW, key_channels)

        key_norm = self.norm(key_features)

        context = key_norm.mean(dim=1)  # (B, key_channels)
        freq_weight = torch.sigmoid(self.context_proj(context)).view(B, self.num_heads, 1)  # (B, num_heads, 1)

        sin, cos = self.rope_2d(H, W, device)

        if key_features.shape[1] > L_spatial:
            pooled_key_features = key_features[:, L_spatial:, :]  # Pooling Feature
            pooled_value_features = value_features[:, L_spatial:, :]

            query_heads_spatial = query_flat.reshape(
                B, H*W, self.num_heads, self.head_dim
            ).permute(0, 2, 1, 3)  # (B, num_heads, HW, head_dim)

            sin_flat = sin.view(H*W, -1)  # (HW, head_dim)
            cos_flat = cos.view(H*W, -1)  # (HW, head_dim)
            query_heads_rope = theta_shift(
                query_heads_spatial, sin_flat.unsqueeze(0).unsqueeze(0), cos_flat.unsqueeze(0).unsqueeze(0)
            )

            query_heads_rope_merged = query_heads_rope.permute(0, 2, 1, 3).reshape(B, H*W, self.key_channels)  # (B, HW, key_channels) # noqa

            attn_pooled = F.softmax(torch.matmul(
                query_heads_rope_merged,  # (B, HW, key_channels)
                pooled_key_features.transpose(1, 2)  # (B, key_channels, num_pooled_features)
            ) * self.scaling, dim=-1)  # (B, HW, num_pooled_features)

            pooled_output = torch.matmul(attn_pooled, pooled_value_features)  # (B, HW, value_channels)

            out_spatial = pooled_output * self.pool_weight  # (B, HW, value_channels)
        else:
            out_spatial = torch.zeros(B, H*W, self.value_channels, device=device)

        query_heads = query_flat.reshape(B, H*W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        key_heads = key_spatial.reshape(B, H*W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        value_heads = value_spatial.reshape(B, H*W, self.num_heads, self.value_head_dim).permute(0, 2, 1, 3)

        out_freq = self.fourier_attention(query_heads, key_heads, value_heads, freq_weight, H, W, sin, cos)
        out_freq = out_freq.permute(0, 2, 1, 3).reshape(B, H*W, self.value_channels)

        output = self.freq_weight * out_freq + out_spatial

        output = output.permute(0, 2, 1).reshape(B, self.value_channels, H, W)
        output = self.ReorganizeConv(output)

        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class HSAModule(nn.Module):
    def __init__(self, in_channels: int = 512, out_channels: int = 512,
                 ratio: int = 2, num_heads: int = 4):
        super().__init__()

        self.spectral_attention = SpectralSpatialAttention(
            in_channels=in_channels,
            out_channels=out_channels,
            ratio=ratio,
            num_heads=num_heads
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        att = self.spectral_attention(x)
        output = att + x
        return output

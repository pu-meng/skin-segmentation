"""Simple bottleneck controls, matched by active trainable parameter count.

The default budget is the whole original V2 block, including its LayerNorm.
No inactive parameters are added to make counts agree.
"""
import torch
from torch import nn


class ParameterMatchedBottleneck(nn.Module):
    def __init__(self, kind, channels=512, target_parameters=1695744, seed=44,
                 heads=8, max_relative_error=0.002):
        super().__init__()
        if kind not in ('mlp', 'conv', 'transformer'):
            raise ValueError(kind)
        if channels % heads and kind == 'transformer':
            raise ValueError('channels must be divisible by heads')
        self.kind = kind
        # Counts include biases, outer normalization, and transformer norm2.
        if kind == 'mlp':
            fixed, slope = 3 * channels, 2 * channels + 1
        elif kind == 'conv':
            fixed, slope = 3 * channels, 18 * channels + 1
        else:
            fixed, slope = 4 * channels**2 + 9 * channels, 2 * channels + 1
        width = max(1, round((target_parameters - fixed) / slope))
        if fixed + slope * width <= 0:
            raise ValueError('Invalid parameter budget')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.norm = nn.LayerNorm(channels)
            if kind == 'conv':
                self.mixer = nn.Sequential(
                    nn.Conv2d(channels, width, 3, padding=1), nn.GELU(),
                    nn.Conv2d(width, channels, 3, padding=1))
            else:
                self.mixer = nn.Sequential(nn.Linear(channels, width), nn.GELU(),
                                           nn.Linear(width, channels))
                if kind == 'transformer':
                    self.attention = nn.MultiheadAttention(channels, heads, dropout=0., batch_first=True)
                    self.norm2 = nn.LayerNorm(channels)
        self.width = width
        self.target_parameters = target_parameters
        self.parameter_count = sum(p.numel() for p in self.parameters())
        assert self.parameter_count == fixed + slope * width
        self.relative_error = (self.parameter_count - target_parameters) / target_parameters
        if abs(self.relative_error) > max_relative_error:
            raise ValueError(f'Cannot match budget within tolerance: {self.parameter_count} vs {target_parameters}')

    def forward(self, feature):
        b, c, h, w = feature.shape
        tokens = feature.flatten(2).transpose(1, 2)
        normalized = self.norm(tokens)
        if self.kind == 'conv':
            spatial = normalized.transpose(1, 2).reshape(b, c, h, w)
            return feature + self.mixer(spatial)
        if self.kind == 'transformer':
            mixed = self.attention(normalized, normalized, normalized, need_weights=False)[0]
            tokens = tokens + mixed
            tokens = tokens + self.mixer(self.norm2(tokens))
        else:
            tokens = tokens + self.mixer(normalized)
        return tokens.transpose(1, 2).reshape(b, c, h, w)

import torch
import torch.nn as nn

from .utils import WTConvBlock, FourierBlock


class F2Block(nn.Module):
    def __init__(self, channels, wt_levels=1, wt_type='db1', spatial_shape=(224, 224)):
        super(F2Block, self).__init__()

        # WTConv
        self.wtconv_branch = WTConvBlock(channels, wt_levels, wt_type)

        # Spectral
        self.spectral_branch = FourierBlock(channels=channels, spatial_shape=spatial_shape)

        # Dynamic Fusion
        self.distance_weight = nn.Parameter(torch.tensor([0.5, 0.5]))

    def forward(self, x):
        identity = x

        wtconv_out = self.wtconv_branch(x)
        spectral_out = self.spectral_branch(x)

        weights = torch.softmax(self.distance_weight, dim=0)

        fused = weights[0] * wtconv_out + weights[1] * spectral_out + identity

        return fused

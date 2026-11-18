#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import torch
from torch import nn
from torch.nn import functional as F


class ConvBNReLU(nn.Module):
    """Conv2d -> BN -> ReLU"""
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SepConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.dw = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride,
                            padding=padding, dilation=dilation, groups=in_channels, bias=False)
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.relu(x)


class MultiScaleBlockLite(nn.Module):
    """Multi-scale block: x + F(x)  (selectively lightweight)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels // 2

        self.branch1 = ConvBNReLU(in_channels, mid_channels, kernel_size=3, stride=2, padding=1)
        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            SepConvBNReLU(mid_channels, mid_channels, kernel_size=3, stride=2, padding=1)
        )

        self.branch3_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.branch3_conv = ConvBNReLU(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)

        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0, bias=False)

        self.fusion = ConvBNReLU(mid_channels * 3, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3_pool(x)
        x3 = self.branch3_conv(x3)

        out = torch.cat([x1, x2, x3], dim=1)
        out = self.fusion(out)
        residual = self.skip(x)
        return out + residual


class Stem(nn.Module):
    def __init__(self, in_channels, mid_channels=(32, 64, 128), edge_channels=4, out_channels=21):
        super().__init__()
        self.block1 = MultiScaleBlockLite(in_channels, mid_channels[0])

        self.block2 = ConvBNReLU(mid_channels[0], mid_channels[1], kernel_size=3, stride=2, padding=1)
        self.block3 = ConvBNReLU(mid_channels[1], mid_channels[2], kernel_size=3, stride=2, padding=1)

        self.sobel = SobelEdge(in_channels)
        self.edge_refine = ConvBNReLU(1, edge_channels)
        self.edge_prob_head = nn.Conv2d(edge_channels, 1, 1, bias=True)

        cat_channels = mid_channels[0] + mid_channels[1] + mid_channels[2] + edge_channels
        self.out_block = ConvBNReLU(cat_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def feature_extract(self, x):
        s1 = self.block1(x)
        s2 = self.block2(s1)
        s3 = self.block3(s2)

        edge0 = self.sobel(x)  # (B,1,H,W)
        edge0 = edge0 / (edge0.mean(dim=(2, 3), keepdim=True) + 1e-6)

        edge_feat = self.edge_refine(edge0)  # (B,edge_ch,H,W)
        edge_prob = torch.sigmoid(self.edge_prob_head(edge_feat))  # (B,1,H,W)
        # print(s1.shape, s2.shape, s3.shape)

        s1 = F.interpolate(s1, size=x.shape[-2:], mode="bilinear", align_corners=False)
        s2 = F.interpolate(s2, size=x.shape[-2:], mode="bilinear", align_corners=False)
        s3 = F.interpolate(s3, size=x.shape[-2:], mode="bilinear", align_corners=False)

        cat_feat = torch.cat([s1, s2, s3, edge_feat], 1)
        return self.out_block(cat_feat), edge_prob

    def forward(self, x):
        feat, edge_prob = self.feature_extract(x)
        return torch.cat([x, feat], 1), edge_prob


class SobelEdge(nn.Module):
    """Differentiable Sobel gradient magnitude."""
    def __init__(self, in_ch: int):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=torch.float32)
        ky = torch.tensor([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=torch.float32)

        self.conv_x = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.conv_y = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        with torch.no_grad():
            self.conv_x.weight[:] = kx.view(1, 1, 3, 3).repeat(in_ch, 1, 1, 1)
            self.conv_y.weight[:] = ky.view(1, 1, 3, 3).repeat(in_ch, 1, 1, 1)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        gx = self.conv_x(x)
        gy = self.conv_y(x)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-12)   # (B,C,H,W)
        mag = mag.mean(dim=1, keepdim=True)           # (B,1,H,W)
        return mag



def pixel_pos_embedding(input_tensor: torch.Tensor,
                        n_freq: int = 2,
                        pos_weight: float = 1.0) -> torch.Tensor:
    """
    Create positional embedding for superpixel input images

    Args:
        input_tensor (torch.Tensor): Input tensor of shape (B, C, H, W)
        n_freq (int): Number of frequency components for embedding
        pos_weight (float): Weight multiplier for the positional embedding

    Returns:
        embedded_x (torch.Tensor): Positional embedding tensor of shape (B, embedding_dim, H, W)
    """
    B, C, H, W = input_tensor.shape
    device = input_tensor.device

    coords = torch.stack(
        torch.meshgrid(
            torch.linspace(-1, 1, H, dtype=torch.float, device=device),
            torch.linspace(-1, 1, W, dtype=torch.float, device=device),
            indexing='ij'
        ),
        -1
    ).unsqueeze(0)  # Shape: (1, H, W, 2)

    freqs = 2 ** torch.arange(n_freq, dtype=torch.float, device=device)  # 指定设备
    shape = coords.shape[:-1] + (-1,)  # (1, H, W, -1)
    scaled_x = (coords[..., None, :] * freqs[..., None]).reshape(shape)  # (1, H, W, 2*n_freq)
    scaled_x = torch.stack([scaled_x, scaled_x + 0.5 * torch.pi], -2).reshape(shape)  # (1, H, W, 4*n_freq)
    embedded_x = torch.sin(scaled_x).permute(0, 3, 1, 2) * pos_weight  # (1, 4*n_freq, H, W)
    embedded_x = embedded_x.expand(B, -1, -1, -1)  # (B, 4*n_freq, H, W)

    return embedded_x



if __name__ == '__main__':
    import time
    x_ = torch.randn(1, 3, 256, 256)
    start_time = time.time()
    model = Stem(in_channels=3, )
    y = model(x_)
    print('time:', time.time() - start_time)
    print(y[0].shape)
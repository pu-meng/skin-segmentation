"""Explicit basic 2D U-Net baseline, base width 32, no pretrained weights."""
import torch
from torch import nn


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )


class UNet2D(nn.Module):
    def __init__(self):
        super().__init__()
        channels = (32, 64, 128, 256, 512)
        self.encoders = nn.ModuleList([
            DoubleConv(3 if i == 0 else channels[i - 1], c)
            for i, c in enumerate(channels)
        ])
        self.pool = nn.MaxPool2d(2)
        self.ups = nn.ModuleList([nn.ConvTranspose2d(channels[i], channels[i - 1], 2, stride=2)
                                 for i in range(4, 0, -1)])
        self.decoders = nn.ModuleList([DoubleConv(channels[i - 1] * 2, channels[i - 1])
                                      for i in range(4, 0, -1)])
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        skips = []
        for i, encoder in enumerate(self.encoders):
            if i:
                x = self.pool(x)
            x = encoder(x)
            skips.append(x)
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips[:-1])):
            x = decoder(torch.cat((up(x), skip), dim=1))
        return torch.sigmoid(self.head(x))

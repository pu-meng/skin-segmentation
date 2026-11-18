"""Local, unverified replacements for the author's missing layers.
The public release omits these components. They are approximations, not author code.
"""
import torch
from torch import nn
import torch.nn.functional as F

class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, drop_out=False):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                  nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
        if drop_out:
            layers.append(nn.Dropout2d(p=0.5))
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)

class UpCat(nn.Module):
    def __init__(self, in_size, out_size, is_deconv=True):
        super().__init__()
        self.up = (nn.ConvTranspose2d(in_size, out_size, 2, stride=2) if is_deconv else
                   nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(in_size, out_size, 1)))
    def forward(self, encoder, decoder):
        up = self.up(decoder)
        if up.shape[-2:] != encoder.shape[-2:]:
            raise ValueError(f'up/skip shape mismatch: {up.shape} vs {encoder.shape}')
        return torch.cat((encoder, up), dim=1)

"""Move the original A1 dual-source skip to one real encoder scale."""
import math
import types
import torch
from torch import nn
from torch.nn import functional as F
from operations.encoder_swap_A016_20261107.encoder_swap_candidate import build_candidate as base_builder


def dynamic_fourier(self, x):
    t = self.n(x)
    h, w = t.shape[-2:]
    with torch.autocast(x.device.type, enabled=False):
        fy = torch.fft.fftfreq(h, device=x.device)[:, None]
        fx = torch.fft.rfftfreq(w, device=x.device)[None, :]
        radius = (fx.square() + fy.square()).sqrt()
        centers = torch.tensor([0., .2, .4, .65], device=x.device)[:, None, None]
        phi = torch.exp(-(radius[None] - centers).square() / (2 * .15 ** 2))
        weight = 1 + .5 * torch.tanh(torch.einsum('cj,jhw->chw', self.a, phi))
        y = torch.fft.irfft2(weight[None] * torch.fft.rfft2(t.float(), norm='ortho'), s=(h, w), norm='ortho')
    return x + self.out(y.to(t.dtype))


class AdaptedSkip(nn.Module):
    def __init__(self, original, channels):
        super().__init__()
        self.original = original
        self.encoder_in = nn.Conv2d(channels, 64, 1)
        self.decoder_in = nn.Conv2d(channels, 64, 1)
        self.encoder_out = nn.Conv2d(64, channels, 1)
        self.decoder_out = nn.Conv2d(64, channels, 1)
        nn.init.zeros_(self.encoder_out.bias)
        nn.init.zeros_(self.decoder_out.bias)
        for side in (original.s, original.d):
            side.gen.fourier.forward = types.MethodType(dynamic_fourier, side.gen.fourier)

    def forward(self, skip, decoder):
        s, d = self.encoder_in(skip), self.decoder_in(decoder)
        s_new, d_new = self.original(s, d)
        return skip + self.encoder_out(s_new - s), decoder + self.decoder_out(d_new - d)


def decoder_forward(self, features):
    x = self.stages[3](self.e2(features[3]))
    for level in (2, 1, 0):
        x = self.ups[level](x)
        skip = features[level]
        if level == self.skip_position:
            skip, x = self.e3(skip, x)
        x = self.stages[level](torch.cat([x, skip], 1))
    return torch.sigmoid(F.interpolate(self.head(x), size=(256, 256), mode='bilinear', align_corners=False))


def build_candidate(position, pretrained=True):
    if position not in (1, 2):
        raise ValueError('Only existing nonbaseline skip1 and skip2 are supported')
    model = base_builder('swiftformer_s__A1', pretrained=pretrained)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(46 + position)
        model.decoder.e3 = AdaptedSkip(model.decoder.e3, {1: 128, 2: 256}[position])
    model.decoder.skip_position = position
    model.decoder.forward = types.MethodType(decoder_forward, model.decoder)
    return model

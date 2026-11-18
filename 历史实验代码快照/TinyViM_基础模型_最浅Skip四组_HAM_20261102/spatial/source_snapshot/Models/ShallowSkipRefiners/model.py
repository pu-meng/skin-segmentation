"""Four separate refiners on baseline aligned S0, immediately before decoder D2 concat."""
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'source/comparison'))
from pretrained_unet2d import PretrainedUNet2D, prepare_imports


class SpatialGate(nn.Module):
    """Decoder-conditioned additive spatial gate; one weight per spatial position."""
    def __init__(self):
        super().__init__()
        self.skip_projection = nn.Conv2d(64,32,1,bias=False)
        self.context_projection = nn.Conv2d(64,32,1,bias=False)
        self.score = nn.Conv2d(32,1,1,bias=True)

    def weights(self, skip, context):
        return self.score(F.relu(self.skip_projection(skip)+self.context_projection(context))).sigmoid()

    def forward(self, skip, context):
        return skip*self.weights(skip,context)


class ChannelGate(nn.Module):
    """SE-style global average -> 64/8/64 channel gating, shared over positions."""
    def __init__(self):
        super().__init__()
        self.down = nn.Conv2d(64,8,1,bias=True)
        self.up = nn.Conv2d(8,64,1,bias=True)

    def weights(self, skip):
        return self.up(F.relu(self.down(skip.mean(dim=(-2,-1),keepdim=True)))).sigmoid()

    def forward(self, skip, context=None):
        return skip*self.weights(skip)


class ConvTransform(nn.Module):
    """One bias-free depthwise3x3 + pointwise1x1; no activation/norm/residual."""
    def __init__(self):
        super().__init__()
        self.depthwise = nn.Conv2d(64,64,3,padding=1,groups=64,bias=False)
        self.pointwise = nn.Conv2d(64,64,1,bias=False)

    def forward(self, skip, context=None):
        return self.pointwise(self.depthwise(skip))


class RegionReconstruction(nn.Module):
    """Single-pass per-image region aggregation and broadcast, K=8, cosine/tau=.1.

    This is a simple diagnostic baseline, not a reproduction of AD-GBC or Slot Attention.
    The learned centers produce assignments; image-specific weighted means are broadcast.
    """
    def __init__(self, num_regions=8, temperature=.1):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_regions,64)*.02)
        self.temperature = temperature

    def aggregate(self, skip):
        tokens = skip.flatten(2).transpose(1,2)
        scores = F.normalize(tokens,dim=-1) @ F.normalize(self.centers,dim=-1).T / self.temperature
        membership = scores.softmax(-1)
        weights = membership / membership.sum(1,keepdim=True).clamp_min(1e-6)
        regions = weights.transpose(1,2) @ tokens
        return tokens, membership, regions

    def forward(self, skip, context=None):
        _, membership, regions = self.aggregate(skip)
        reconstructed = membership @ regions
        return reconstructed.transpose(1,2).reshape_as(skip)


REFINERS = {'spatial':SpatialGate, 'channel':ChannelGate, 'conv':ConvTransform, 'region':RegionReconstruction}


class TinyViMShallowSkipRefiner(PretrainedUNet2D):
    def __init__(self, kind, pretrained=True):
        if kind not in REFINERS:
            raise ValueError(kind)
        prepare_imports('mambavision_t')
        super().__init__('tinyvim_s',pretrained=pretrained,new_layers_seed=42)
        self.kind = kind
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(44)
            self.skip_refiner = REFINERS[kind]()

    def decode_features(self, features, refine=True):
        x = features[-1]
        skips = list(reversed(features[:-1]))
        for i, (up,block) in enumerate(zip(self.decoder.ups,self.decoder.blocks)):
            x = up(x)
            if i < 3:
                skip = skips[i]
                if i == 2 and refine:
                    skip = self.skip_refiner(skip,x)
                x = torch.cat((x,skip),dim=1)
            x = block(x)
        return torch.sigmoid(self.decoder.head(x))

    def forward(self, x):
        features = [align(f) for align,f in zip(self.align,self.raw_features(x))]
        return self.decode_features(features)

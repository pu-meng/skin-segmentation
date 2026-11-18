"""TinyViM-S encoder with the trained V5 UNet decoder and unchanged state keys."""

import torch
from torch import nn

CHANNELS = (48, 64, 168, 224)


def official_backbone():
    from .backbone import TinyViM_S
    backbone = TinyViM_S(pretrained=False)
    backbone.head = nn.Identity()
    if hasattr(backbone, 'dist_head'):
        backbone.dist_head = nn.Identity()
    return backbone


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = official_backbone()

    def forward(self, x):
        features, handles = [], []
        try:
            for index in (0, 2, 4):
                handles.append(self.backbone.network[index].register_forward_hook(
                    lambda module, args, output: features.append(output)))
            x = self.backbone.patch_embed(x)
            x = self.backbone.forward_tokens(x)
            features.append(self.backbone.norm(x))
        finally:
            for handle in handles:
                handle.remove()
        expected = [(c, 256 // (4 * 2**i), 256 // (4 * 2**i))
                    for i, c in enumerate(CHANNELS)]
        if [tuple(f.shape[1:]) for f in features] != expected:
            raise ValueError('Unexpected TinyViM feature shapes')
        return features


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        channels = (512, 256, 128, 64, 32, 16)
        self.ups = nn.ModuleList([nn.ConvTranspose2d(a, b, 2, stride=2)
                                 for a, b in zip(channels[:-1], channels[1:])])
        self.blocks = nn.ModuleList([DoubleConv(c * (2 if i < 3 else 1), c)
                                     for i, c in enumerate(channels[1:])])
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, features):
        x, skips = features[-1], list(reversed(features[:-1]))
        for i, (up, block) in enumerate(zip(self.ups, self.blocks)):
            x = up(x)
            if i < 3:
                x = torch.cat((x, skips[i]), dim=1)
            x = block(x)
        return torch.sigmoid(self.head(x))


class TinyViMUNet(nn.Module):
    """RGB [0,1], BCHW 256x256 -> foreground probabilities; all layers trainable.

    Constructor initializes weights only. Load a complete segmentation checkpoint
    explicitly with load_checkpoint(); this avoids confusing ImageNet pretraining
    with the already trained segmentation model.
    """
    def __init__(self, new_layers_seed=42):
        super().__init__()
        self.encoder = Encoder()
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(new_layers_seed)
            self.decoder = Decoder()
            torch.manual_seed(new_layers_seed + 1)
            self.align = nn.ModuleList([nn.Conv2d(a, b, 1) for a, b in
                                       zip(CHANNELS, (64, 128, 256, 512))])

    def raw_features(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, 256, 256):
            raise ValueError('Expected RGB [0,1] input [B,3,256,256]')
        return self.encoder((x - self.mean) / self.std)

    def forward(self, x):
        return self.decoder([a(f) for a, f in zip(self.align, self.raw_features(x))])

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        state = checkpoint.get('model', checkpoint)
        self.load_state_dict(state, strict=True)
        return self

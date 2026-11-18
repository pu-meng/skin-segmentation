"""Official RMT classification trunk and the existing shared segmentation decoder."""
import importlib.util
import sys
from pathlib import Path

import torch
from torch import nn

from pretrained_unet2d import SharedDecoder, sha256_file

RMT_ROOT = Path('/home/PuMengYu/experiments/RMT')
SOURCE = RMT_ROOT / 'official_code/classfication_release/RMT.py'
VARIANTS = {
    'rmt_t': ('RMT_T3', 'RMT-T.pth', '6196c191d82d02434da90fc09b7003316ed093b02aeb05be5a8a07fdfcbe800c'),
    'rmt_s': ('RMT_S', 'RMT-S.pth', '0da6ae9e7dadd6b67fd57f1543f01b5220e34676ea45c8bb703881e3414edb1d'),
}


def official_model(variant):
    module_name = 'ham_official_rmt'
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, SOURCE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
    return getattr(sys.modules[module_name], VARIANTS[variant][0])(None)


class RMTEncoder(nn.Module):
    def __init__(self, variant, pretrained=True):
        super().__init__()
        self.backbone = official_model(variant)
        self.loading_audit = None
        if pretrained:
            _, filename, expected = VARIANTS[variant]
            path = RMT_ROOT / 'weights' / filename
            if sha256_file(path) != expected:
                raise ValueError('Uploaded weight hash changed: ' + str(path))
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            source = checkpoint['model']
            self.backbone.load_state_dict(source, strict=True)
            for name, tensor in self.backbone.state_dict().items():
                if not torch.equal(tensor, source[name]):
                    raise ValueError('Loaded tensor differs: ' + name)
            self.loading_audit = {
                'variant': variant, 'weights_path': str(path), 'sha256': expected,
                'bytes': path.stat().st_size, 'checkpoint_branch': 'model',
                'model_ema_used': False, 'full_classifier_strict_load': True,
                'all_loaded_tensors_equal': True, 'missing': [], 'unexpected': [],
                'full_classifier_state_entries': len(source),
                'removed_classification_modules': ['proj', 'norm', 'swish', 'avgpool', 'head'],
                'feature_path': 'patch_embed then four layers; capture pre-downsample BHWC features and convert to BCHW',
                'extra_feature_norms': False,
            }
        # The projection/BN/swish/pooling/head are all classification neck/head.
        # Dense features come directly from the four pretrained stages.
        for name in ('proj', 'norm', 'swish', 'avgpool', 'head'):
            setattr(self.backbone, name, nn.Identity())

    def forward(self, x):
        features, handles = [], []
        try:
            for layer in self.backbone.layers[:3]:
                handles.append(layer.downsample.register_forward_pre_hook(
                    lambda module, args: features.append(args[0].permute(0, 3, 1, 2).contiguous())))
            x = self.backbone.patch_embed(x)
            for layer in self.backbone.layers:
                x = layer(x)
            features.append(x.permute(0, 3, 1, 2).contiguous())
        finally:
            for handle in handles:
                handle.remove()
        expected = [(64, 64, 64), (128, 32, 32), (256, 16, 16), (512, 8, 8)]
        if [tuple(f.shape[1:]) for f in features] != expected:
            raise ValueError('RMT feature dimensions differ from the shared interface')
        return features


class RMTUNet2D(nn.Module):
    def __init__(self, variant, pretrained=True, new_layers_seed=42):
        super().__init__()
        self.encoder = RMTEncoder(variant, pretrained)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(new_layers_seed)
            self.decoder = SharedDecoder()
            torch.manual_seed(new_layers_seed + 1)
            self.align = nn.ModuleList([nn.Conv2d(c, c, 1) for c in (64, 128, 256, 512)])

    def raw_features(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, 256, 256):
            raise ValueError('Expected RGB /255 [B,3,256,256]')
        return self.encoder((x - self.mean) / self.std)

    def forward(self, x):
        return self.decoder([a(f) for a, f in zip(self.align, self.raw_features(x))])

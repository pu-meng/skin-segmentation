"""Three official pretrained encoders with the agreed shared five-stage decoder.

Upstream repositories and baseline UNet2D are never edited. Inputs are RGB /255.
"""
import hashlib
import importlib
import importlib.util
import importlib.metadata
import json
import sys
import types
from pathlib import Path

import torch
from torch import nn

from unet2d import DoubleConv


MATERIALS = Path('/home/PuMengYu/研究存档/20261020_论文汇总/方案1_问题定义与思路检索/候选编码器_ImageNet1K')
VARIANTS = {
    'fastervit0': ('FasterViT', 'fastervit.models.faster_vit_any_res', 'faster_vit_0_any_res', (64, 128, 256, 512)),
    'mambavision_t': ('MambaVision', 'mambavision.models.mamba_vision', 'mamba_vision_T', (80, 160, 320, 640)),
    'tinyvim_s': ('TinyViM', 'model.tinyvim', 'TinyViM_S', (48, 64, 168, 224)),
}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare_imports(variant):
    """Compatibility imports only; keep the installed numerical operators intact."""
    folder = VARIANTS[variant][0]
    code = str(MATERIALS / folder / 'code')
    if code not in sys.path:
        sys.path.insert(0, code)
    if variant == 'tinyvim_s':
        # timm >=1 moved helpers; this is the same to_2tuple implementation.
        import timm.layers.helpers
        sys.modules.setdefault('timm.models.layers.helpers', timm.layers.helpers)
    if variant == 'mambavision_t' and 'mamba_ssm' not in sys.modules:
        # mamba_ssm.__init__ imports an unrelated language model that requires old
        # transformers. Load the real installed ops submodule without that export.
        spec = importlib.util.find_spec('mamba_ssm')
        if spec is None:
            raise ImportError('Installed mamba_ssm is required')
        package = types.ModuleType('mamba_ssm')
        package.__path__ = list(spec.submodule_search_locations)
        package.__spec__ = spec
        package.__package__ = 'mamba_ssm'
        package.__version__ = importlib.metadata.version('mamba-ssm')
        sys.modules['mamba_ssm'] = package
        # Preserve real vision/baseline exports, omitting only language-model
        # imports. Neither the operators nor their autograd are replaced.
        from mamba_ssm.modules.mamba_simple import Mamba
        from mamba_ssm.modules.mamba2 import Mamba2
        package.Mamba, package.Mamba2 = Mamba, Mamba2


def make_official(variant, resolution=256):
    if variant not in VARIANTS:
        raise ValueError(f'Unsupported active encoder: {variant}')
    prepare_imports(variant)
    _, module, factory, _ = VARIANTS[variant]
    kwargs = {'resolution': [resolution, resolution]} if variant == 'fastervit0' else {}
    return getattr(importlib.import_module(module), factory)(pretrained=False, **kwargs)


def material_record(variant):
    folder = VARIANTS[variant][0]
    data = json.loads((MATERIALS / '材料准备状态.json').read_text())
    return next(r for r in data['candidates'] if r['name'] == folder)


def load_official_weights(model, variant):
    """Strict full-classifier validation, followed by explicit geometry conversion."""
    record = material_record(variant)
    path = Path(record['weights_path'])
    if path.stat().st_size != record['bytes'] or sha256_file(path) != record['sha256']:
        raise ValueError(f'Weight integrity failure: {path}')
    checkpoint = torch.load(path, map_location='cpu')
    branch = 'model' if variant == 'tinyvim_s' else 'state_dict'
    source = checkpoint[branch]  # Explicit choice: TinyViM model, not model_ema.
    classifier_reference = None
    if variant == 'fastervit0':
        from fastervit.models.faster_vit import faster_vit_0_224
        classifier_reference = faster_vit_0_224(pretrained=False)
        classifier_reference.load_state_dict(source, strict=True)
    target = model.state_dict()
    if set(source) != set(target):
        raise ValueError({'missing': sorted(set(target) - set(source)),
                          'unexpected': sorted(set(source) - set(target))})
    params = dict(model.named_parameters())
    converted = {}
    mapping = []
    regenerated = []
    for key, value in target.items():
        src = source[key]
        if src.shape != value.shape:
            if (variant != 'fastervit0' or key in params or not key.endswith(
                    ('.relative_coords_table', '.relative_position_index', '.relative_bias'))):
                raise ValueError(f'Unexplained shape mismatch: {key}')
            converted[key] = value  # Official any-res constructor's 256 geometry.
            regenerated.append(key)
            action = 'official_256_geometry_regenerated'
        else:
            converted[key] = src
            action = 'copied_exactly'
        mapping.append({'source': key, 'target': key, 'source_shape': list(src.shape),
                        'target_shape': list(value.shape), 'kind': 'parameter' if key in params else 'buffer',
                        'action': action})
    model.load_state_dict(converted, strict=True)
    for key, value in model.state_dict().items():
        if not torch.equal(value, converted[key]):
            raise AssertionError(f'Loaded tensor mismatch: {key}')
    audit = {'variant': variant, 'source': record, 'checkpoint_branch': branch,
             'full_classifier_strict_load': True, 'missing': [], 'unexpected': [],
             'unexplained_shape_mismatches': [], 'tensor_mapping': mapping,
             'regenerated_geometry_buffers': regenerated,
             'all_loaded_tensors_equal': True}
    return audit, classifier_reference


class OfficialEncoder(nn.Module):
    def __init__(self, variant, pretrained=True):
        super().__init__()
        self.variant = variant
        self.backbone = make_official(variant)
        self.loading_audit = None
        if pretrained:
            self.loading_audit, _ = load_official_weights(self.backbone, variant)
        # Keep final norm in the feature path; only classification heads are removed.
        self.backbone.head = nn.Identity()
        if hasattr(self.backbone, 'dist_head'):
            self.backbone.dist_head = nn.Identity()
        self.nonpersistent_cache_buffers = []
        if variant == 'fastervit0':
            # Upstream forward changes cache shapes. They are recomputed on each
            # call and are not model state; omitting them permits strict reload.
            for name, module in self.backbone.named_modules():
                if 'relative_bias' in module._buffers:
                    module._non_persistent_buffers_set.add('relative_bias')
                    self.nonpersistent_cache_buffers.append(name + '.relative_bias')

    def forward(self, x):
        features = []
        handles = []
        try:
            if self.variant == 'tinyvim_s':
                for index in (0, 2, 4):
                    handles.append(self.backbone.network[index].register_forward_hook(
                        lambda module, args, output: features.append(output)))
                x = self.backbone.patch_embed(x)
                x = self.backbone.forward_tokens(x)
                features.append(self.backbone.norm(x))
            else:
                for level in self.backbone.levels[:3]:
                    handles.append(level.downsample.register_forward_pre_hook(
                        lambda module, args: features.append(args[0])))
                x = self.backbone.patch_embed(x)
                for level in self.backbone.levels:
                    x = level(x)
                features.append(self.backbone.norm(x))
        finally:
            for handle in handles:
                handle.remove()
        expected = [(c, 256 // (4 * 2**i), 256 // (4 * 2**i))
                    for i, c in enumerate(VARIANTS[self.variant][3])]
        shapes = [tuple(f.shape[1:]) for f in features]
        if shapes != expected:
            raise ValueError(f'Unexpected {self.variant} features: {shapes}, expected {expected}')
        return features


class SharedDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        channels = (512, 256, 128, 64, 32, 16)
        self.ups = nn.ModuleList([nn.ConvTranspose2d(a, b, 2, stride=2)
                                 for a, b in zip(channels[:-1], channels[1:])])
        self.blocks = nn.ModuleList([DoubleConv(c * (2 if i < 3 else 1), c)
                                     for i, c in enumerate(channels[1:])])
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, features):
        x = features[-1]
        skips = list(reversed(features[:-1]))
        for i, (up, block) in enumerate(zip(self.ups, self.blocks)):
            x = up(x)
            if i < 3:
                x = torch.cat((x, skips[i]), dim=1)
            x = block(x)
        return torch.sigmoid(self.head(x))


class PretrainedUNet2D(nn.Module):
    def __init__(self, variant, pretrained=True, new_layers_seed=42):
        super().__init__()
        self.encoder = OfficialEncoder(variant, pretrained=pretrained)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        # Separate streams: decoder initialization is exactly the same for every
        # encoder regardless of how much randomness backbone construction used.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(new_layers_seed)
            self.decoder = SharedDecoder()
            torch.manual_seed(new_layers_seed + 1)
            self.align = nn.ModuleList([nn.Conv2d(a, b, 1) for a, b in
                                       zip(VARIANTS[variant][3], (64, 128, 256, 512))])

    def raw_features(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, 256, 256):
            raise ValueError('Expected RGB /255 input [B,3,256,256]')
        return self.encoder((x - self.mean) / self.std)

    def forward(self, x):
        return self.decoder([align(f) for align, f in zip(self.align, self.raw_features(x))])

"""Full official SU-RMT, adapted only for the locked HAM10000 V5 interface."""
import contextlib
import importlib
import importlib.util
import io
import sys
from pathlib import Path

import torch
from torch import nn

from pretrained_unet2d import sha256_file

from project_paths import PROJECT, PATHS
HOME = PROJECT / 'references/code/SU-RMT'
OFFICIAL = HOME
WEIGHTS = Path(PATHS['pretrained_weights']['rmt_t'])
WEIGHT_SHA256 = '6196c191d82d02434da90fc09b7003316ed093b02aeb05be5a8a07fdfcbe800c'
COMMIT = '03d35576ddc3f88a1b1002cdd3b5742641ca6165'


def official_modules():
    package = 'ham_official_surmt'
    if package not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            package, OFFICIAL / 'Models/__init__.py',
            submodule_search_locations=[str(OFFICIAL / 'Models')])
        module = importlib.util.module_from_spec(spec)
        sys.modules[package] = module
        spec.loader.exec_module(module)
    return (importlib.import_module(package + '.SURMT.Networks'),
            importlib.import_module(package + '.SURMT.RMT'))


class SURMTV5(nn.Module):
    def __init__(self, pretrained=True, input_size=256):
        super().__init__()
        if input_size not in (224, 256):
            raise ValueError('Only the paper/reference and V5 sizes are audited')
        networks, rmt = official_modules()
        original_factory, original_f2 = networks.DynamicRMT, networks.F2Block
        audit = {}

        def encoder_factory(*args, **kwargs):
            model = rmt.DynamicRMT(pretrained=False)
            if pretrained:
                if sha256_file(WEIGHTS) != WEIGHT_SHA256:
                    raise ValueError('RMT-T weight hash mismatch')
                source = torch.load(WEIGHTS, map_location='cpu', weights_only=False)['model']
                result = model.load_state_dict(source, strict=False)
                expected_missing = {f'layers.{i}.Relpos.{n}' for i in range(4)
                                    for n in ('distance_weight', 'direction_weight')}
                if set(result.missing_keys) != expected_missing or result.unexpected_keys:
                    raise RuntimeError('Unexpected official encoder loading mismatch: ' + str(result))
                loaded = model.state_dict()
                if any(not torch.equal(loaded[n], t) for n, t in source.items()):
                    raise RuntimeError('Pretrained tensor equality check failed')
                audit.update(weights_path=str(WEIGHTS), sha256=WEIGHT_SHA256,
                             checkpoint_branch='model', model_ema_used=False,
                             loaded_entries=len(source), all_loaded_tensors_equal=True,
                             missing_keys=sorted(expected_missing), unexpected_keys=[],
                             missing_reason='New SU-RMT learnable spatial-prior parameters')
            return model

        def f2_factory(*args, **kwargs):
            kwargs['spatial_shape'] = (input_size // 4, input_size // 4)
            return original_f2(*args, **kwargs)

        # Apply process-local construction adapters; tracked upstream files stay unchanged.
        networks.DynamicRMT, networks.F2Block = encoder_factory, f2_factory
        construction_log = io.StringIO()
        try:
            with contextlib.redirect_stdout(construction_log):
                self.network = networks.SURMT(classes=1, channels=3)
        finally:
            networks.DynamicRMT, networks.F2Block = original_factory, original_f2
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.input_size = input_size
        self.loading_audit = dict(audit, pretrained=pretrained, official_commit=COMMIT,
            official_code_path=str(OFFICIAL), construction_log=construction_log.getvalue(),
            encoder_depths=[2, 2, 8, 2], decoder_depths_deep_to_shallow=[3, 6, 4, 3],
            channels=[64, 128, 256, 512], decoder_initialization='Official load_from_encoder',
            adaptations={'input_size': input_size, 'classes': 1,
                         'f2_spatial_shape': [input_size // 4] * 2,
                         'normalization': 'ImageNet once inside full-model adapter',
                         'output': 'sigmoid of official single-channel segmentation logits'},
            architecture='Full official DySA encoder + HSA + F2 + native dynamic-RMT decoder/head')

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, self.input_size, self.input_size):
            raise ValueError('Expected RGB /255 BCHW at the configured spatial size')
        return torch.sigmoid(self.network((x - self.mean) / self.std))

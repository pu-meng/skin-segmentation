"""Official 2026 encoders with the unchanged V5 shared UNet decoder.

Source checkouts are read-only; strict full-classifier loading precedes head removal.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import torch
from torch import nn
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT/'source/comparison'))
from pretrained_unet2d import SharedDecoder
SPECS = {
    'lowformer_e1': ('LowFormer', 'lowformer_model.py', 'E1.yaml',
                     'LowFormer_E1_evalmodel.pt', (40, 80, 160, 320)),
    'cpubone_b0_bftraining': ('CPUBone', 'cpubone_model.py', 'cpubone_b0_bftraining.yaml',
                            'CPUBone_B0_bftraining_evalmodel.pt', (32, 64, 128, 256)),
    'visionhope_tiny': ('VisionHOPE', None, None, 'visionhope_tiny.pth', (64, 128, 256, 512)),
}
OUT = PROJECT/'references/核对记录/2026_encoder'
WEIGHTS = PROJECT/'Models/pretrained/2026_encoder'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def classifier(name, pretrained=True):
    repo, source, config, filename, channels = SPECS[name]
    root = PROJECT/'references/code'/repo
    path = WEIGHTS/filename
    if pretrained and not path.is_file():
        raise FileNotFoundError('Official weights not downloaded: '+str(path))
    if source:
        spec = importlib.util.spec_from_file_location('official_'+name, root/source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cfg = module.setup_exp_config(str(root/'configs/cls/imagenet'/config), recursive=True, opt_args=None)
        factory = module.lowformer_cls_b1 if repo == 'LowFormer' else module.cpubone_cls_b1
        model = factory(**cfg['net_config'])
        if not pretrained:
            return model, channels, {'pretrained':False}
        if repo == 'CPUBone':
            state = module.remap_legacy_state_dict(module.load_state_dict_from_file(str(path), use_ema=True))
        else:
            state = module.load_state_dict_from_file(str(path))
            state = {k.replace('head.opseq.', 'head.'):v for k,v in state.items()}
    else:
        if 'visionhope' not in sys.modules:
            spec = importlib.util.spec_from_file_location('visionhope', root/'__init__.py',
                                                         submodule_search_locations=[str(root)])
            module = importlib.util.module_from_spec(spec)
            sys.modules['visionhope'] = module
            spec.loader.exec_module(module)
        from visionhope.models import create_model
        from visionhope.tools.checkpoint import read_weights
        model = create_model(name)
        if not pretrained:
            return model, channels, {'pretrained':False}
        state = read_weights(str(path))
    manifest = json.loads((OUT/'materials_and_results.json').read_text())
    expected = next(x for x in manifest['results'] if x['name']==name)
    if path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
        raise ValueError('Official weight integrity differs: '+str(path))
    model.load_state_dict(state, strict=True)
    assert all(torch.equal(v, state[k]) for k,v in model.state_dict().items())
    audit = {'repo':str(root), 'commit':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
             'weights':str(path), 'bytes':path.stat().st_size, 'sha256':digest(path),
             'classification_parameters':sum(v.numel() for v in model.parameters()),
             'full_classifier_strict_load':True, 'all_loaded_tensors_equal':True,
             'weight_branch':'EMA first (official reader)' if repo=='CPUBone' else
                             ('state_dict (official reader)' if repo=='VisionHOPE' else 'official state_dict reader'),
             'source_sha256':{str(f.relative_to(root)):digest(f) for f in root.rglob('*.py')},
             'config_sha256':digest(root/'configs/cls/imagenet'/config) if config else None}
    return model, channels, audit


class Encoder(nn.Module):
    def __init__(self, model, name):
        super().__init__()
        self.name = name
        if name.startswith('visionhope'):
            self.backbone = model
            # forward_intermediates uses none of these classification-only modules.
            for key in ('norm', 'head_pre_logits', 'head_norm', 'head_act', 'head'):
                setattr(self.backbone, key, nn.Identity())
        else:
            self.backbone = model.backbone

    def forward(self, x):
        if self.name.startswith('visionhope'):
            return self.backbone.forward_intermediates(x)
        features = self.backbone(x)
        return [features[key] for key in ('stage1','stage2','stage3','stage_final')]


class EncoderUNet2026(nn.Module):
    def __init__(self, encoder, channels):
        super().__init__()
        self.encoder = encoder
        self.register_buffer('mean',torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('std',torch.tensor([.229,.224,.225]).view(1,3,1,1))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            self.decoder = SharedDecoder()
            torch.manual_seed(43)
            self.align = nn.ModuleList([nn.Conv2d(a,b,1) for a,b in zip(channels,(64,128,256,512))])

    def forward(self, x):
        return self.decoder([layer(f) for layer,f in zip(self.align,self.encoder((x-self.mean)/self.std))])



def build_model(name, pretrained=True):
    if name not in ('lowformer_e1', 'cpubone_b0_bftraining'):
        raise ValueError('Only the two approved 2026 encoders are enabled')
    model, channels, audit = classifier(name, pretrained=pretrained)
    result = EncoderUNet2026(Encoder(model,name), channels)
    result.encoder.loading_audit = audit
    return result

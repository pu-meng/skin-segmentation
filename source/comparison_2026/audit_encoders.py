"""Isolated 2026 encoder material and shared-decoder smoke audit; no dataset/train.

Official checkouts and the established comparison code are not modified.
Classification weights must load strictly before heads are removed.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

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


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def classifier(name):
    repo, source, config, filename, channels = SPECS[name]
    root = PROJECT/'references/code'/repo
    path = WEIGHTS/filename
    if not path.is_file():
        raise FileNotFoundError('Official weights not downloaded: '+str(path))
    if source:
        spec = importlib.util.spec_from_file_location('official_'+name, root/source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cfg = module.setup_exp_config(str(root/'configs/cls/imagenet'/config), recursive=True, opt_args=None)
        factory = module.lowformer_cls_b1 if repo == 'LowFormer' else module.cpubone_cls_b1
        model = factory(**cfg['net_config'])
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
        state = read_weights(str(path))
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


class SharedUNet(nn.Module):
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


def run(name, device):
    torch.manual_seed(42)
    model, channels, report = classifier(name)
    report.update(name=name, status='strict_weights_loaded', device=str(device),
                  test_data_accessed=False, formal_training_started=False,
                  torch_version=torch.__version__, batch=1, input=[1,3,256,256])
    write(OUT/(name+'.json'), report)
    encoder = Encoder(model, name)
    del model
    segmentation = SharedUNet(encoder, channels).to(device).eval()
    x = torch.rand(1,3,256,256,device=device)
    with torch.no_grad():
        features = encoder((x-segmentation.mean)/segmentation.std)
    shapes = [list(f.shape) for f in features]
    assert shapes == [[1,c,256//(4*2**i),256//(4*2**i)] for i,c in enumerate(channels)], shapes
    report.update(feature_shapes=shapes, encoder_parameters=sum(p.numel() for p in encoder.parameters()),
                  full_segmentation_parameters=sum(p.numel() for p in segmentation.parameters()),
                  shared_decoder_unchanged=True)
    write(OUT/(name+'.json'), report)
    del features
    optimizer = torch.optim.AdamW(segmentation.parameters(),lr=1e-4)
    prediction = segmentation(x)
    assert tuple(prediction.shape)==(1,1,256,256) and torch.isfinite(prediction).all()
    loss = (prediction-.5).square().mean()
    loss.backward()
    missing = [n for n,p in segmentation.named_parameters() if p.grad is None]
    assert not missing, missing
    assert all(torch.isfinite(p.grad).all() for p in segmentation.parameters())
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        expected = segmentation(x).detach().cpu()
    temp = OUT/(name+'_temporary_smoke.pth')
    try:
        torch.save(segmentation.state_dict(),temp)
        segmentation.load_state_dict(torch.load(temp,map_location=device,weights_only=True), strict=True)
        with torch.no_grad():
            actual = segmentation(x).detach().cpu()
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
    finally:
        temp.unlink(missing_ok=True)
    report.update(status='passed', output=list(prediction.shape), forward_backward=True,
                  optimizer_step=True, missing_gradients=[], gradients_finite=True,
                  strict_checkpoint_roundtrip=True, completed_at=datetime.now().astimezone().isoformat())
    write(OUT/(name+'.json'),report)
    print(name,'PASSED',shapes,report['full_segmentation_parameters'],flush=True)
    del segmentation,encoder,optimizer,prediction,x
    gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',choices=list(SPECS),required=True)
    parser.add_argument('--device',default='cpu')
    args = parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    try:
        run(args.model,torch.device(args.device))
    except Exception as error:
        path = OUT/(args.model+'.json')
        report = json.loads(path.read_text()) if path.exists() else {'name':args.model}
        report.update(status='failed',error=repr(error),formal_training_started=False)
        write(path,report)
        raise

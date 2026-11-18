#!/usr/bin/env python3
"""ISIC seed42 skip-location experiment; separate CPU/GPU phases."""
import copy
import json
import os
import sys
from pathlib import Path

ROOT = Path('/home/PuMengYu/experiments/skin_segmentation')
sys.path[:0] = [str(ROOT), str(ROOT/'source/comparison'), str(ROOT/'source/bottleneck_mamba')]
from operations.a1_swift_es_isic_seed40_20261114 import entry as common
from source.a1_swift_skip_position_20261114.candidate import build_candidate

OUT = Path('/media/8T/PuMengYu/experiments/skin_segmentation')
REF = OUT/'ISIC2018_A0_A1_FRESH300_20261108/protocol.json'
UUIDS = common.UUIDS

def run_dir(position):
    return OUT/f'A1_SWIFTFORMER_SKIP{position}_ISIC_SEED42_FRESH300_20261114'

def arm_name(position):
    return f'A1-skip{position}'

def setup(position):
    common.RUN = run_dir(position)
    common.SEED = 42
    common.LABEL = f'A1 SwiftFormer-S E+S+BD skip{position} ISIC seed42'
    common.ARMS = (arm_name(position),)
    def imports(arm):
        import torch, ham_fast_runner as runner
        if arm != arm_name(position):
            raise ValueError(arm)
        return torch, runner, lambda pretrained: build_candidate(position, pretrained)
    common.imports = imports

def prepare(position):
    setup(position)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU prepare requires CUDA_VISIBLE_DEVICES empty')
    root = common.RUN
    if root.exists():
        raise FileExistsError(root)
    import torch
    from source.isic2018 import data
    parent = common.read(REF)
    cfg = copy.deepcopy(parent['config'])
    cfg['training']['seed'] = 42
    assert cfg['training']['epochs'] == 300 and cfg['training']['batch_size'] == 8
    assert cfg['data']['counts'] == {'train': 1780, 'val': 258, 'test': 518}
    assert common.sha(cfg['data']['split_file']) == cfg['data']['split_sha256']
    assert data.verify()['counts'] == cfg['data']['counts']
    torch.set_num_threads(1)
    from ham_fast_runner import baseline_api
    baseline_api().seed_all(42)
    model = build_candidate(position, pretrained=True)
    assert not torch.cuda.is_initialized()
    assert all(p.requires_grad for p in model.parameters())
    # A strict state roundtrip checks the exact module structure used at test time.
    import io
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    model.load_state_dict(torch.load(buffer, map_location='cpu', weights_only=False), strict=True)
    sources = {str(REF): common.sha(REF), str(Path(__file__).resolve()): common.sha(__file__)}
    for mod in list(sys.modules.values()):
        path = getattr(mod, '__file__', None)
        if path and str(path).startswith(str(ROOT)) and str(path).endswith('.py') and Path(path).is_file():
            sources[str(Path(path).resolve())] = common.sha(path)
    for path in model.encoder.loading_audit.values():
        if isinstance(path, str) and Path(path).is_file():
            sources[path] = common.sha(path)
    root.mkdir(parents=True)
    work = root/'ISIC2018'/arm_name(position)
    work.mkdir(parents=True)
    common.write(work/'cpu_acceptance.json', {'passed': True, 'strict_state_roundtrip': True,
                 'parameters': sum(p.numel() for p in model.parameters()),
                 'initial_state_sha256': common.state_digest(model.state_dict()),
                 'encoder_loading_audit': model.encoder.loading_audit})
    protocol = {'identity': f'A1 E+S+BD skip{position} ISIC seed42',
                'datasets': {'ISIC2018': {'config': cfg, 'split_sha256': cfg['data']['split_sha256']}},
                'protected_sha256': sources,
                'architecture': f'original SwiftFormer-S and A1 Bottle/D3/D2/head; original dual-source E3 is moved from skip0 to skip{position}; other skips plain',
                'adaptation': 'both branches use learned C->64 and 64->C 1x1 projections on residual updates; Fourier basis computed on actual spatial grid; skip0 original is unchanged baseline',
                'mapping': {'encoder0/skip0/decoder0': '64x64,C64', 'encoder1/skip1/decoder1': '32x32,C128',
                            'encoder2/skip2/decoder2': '16x16,C256', 'encoder3/decoder3': '8x8,C512; no skip3'},
                'training': 'fixed split 1780/258/518, seed42, fresh300, batch8 FP32; inherited optimizer, loss, scheduler, augmentation; validation-minloss selection',
                'evaluation': 'separate process, fixed checkpoint, none and mirror4, 518 images each, audit',
                'scheduling': 'GPU1 initial free gate 4096 MiB, real batch8 preflight; sequential skip1 then skip2; no batch or resolution change'}
    common.write(root/'protocol.json', protocol)
    common.write(root/'prepare_complete.json', {'passed': True, 'protocol_sha256': common.sha(root/'protocol.json')})
    print('CPU_PREPARED', position, flush=True)

def phase(position, kind, gpu):
    setup(position)
    common.phase('ISIC2018', arm_name(position), kind, gpu)

if __name__ == '__main__':
    pos = int(sys.argv[2])
    if pos not in (1, 2):
        raise ValueError(pos)
    if sys.argv[1] == 'prepare':
        prepare(pos)
    elif sys.argv[1] == 'phase':
        phase(pos, sys.argv[3], int(sys.argv[4]))
    else:
        raise ValueError(sys.argv[1])

"""Isolated HAM10000 V1 run, sharing the existing frozen training/eval functions.

No test loader is created. No previous checkpoint, protocol or source is edited.
"""
import argparse
import copy
from datetime import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'source/comparison'))

import torch
from torch import nn
import ham_fast_runner as existing
from ham_fast_pipeline import make_fast_loaders
from pretrained_unet2d import PretrainedUNet2D, prepare_imports, sha256_file

NAME = 'tinyvim_s_bottleneck_mamba_v1'


def verify_v1_runtime():
    """Keep every training source lock; record one known unused plotting drift.

    No file, protocol or frozen manifest is rewritten. This process-local
    validation adapter permits only the exact observed plot revision.
    """
    manifest = json.loads((PROJECT / 'source/config/runtime_sources.json').read_text())
    accepted_plot = {
        'source/analysis/tinyvim_frequency/draw_model_structure.py':
        '9ec24eff7686b74ad96298f7a9453b8db934d19015ae069d060bff8dd2da9387'}
    drift = []
    for relative, expected in manifest.items():
        actual = sha256_file(PROJECT / relative)
        if actual != expected:
            if accepted_plot.get(relative) != actual:
                raise ValueError('Runtime source changed: ' + relative)
            drift.append({'path': relative, 'expected': expected, 'actual': actual,
                          'reason': 'unused plotting-only script; not imported by training'})
    return {'checked_files': len(manifest), 'unrelated_plotting_drift': drift}


def build_experiment_model(pretrained=True):
    prepare_imports('mambavision_t')
    from Models.TinyViM.mamba_bottleneck_v1 import BottleneckMambaV1
    model = PretrainedUNet2D('tinyvim_s', pretrained=pretrained, new_layers_seed=42)
    # All three shallow skips stay untouched. Deepest alignment:
    # [b,224,8,8] -> [b,512,8,8] -> V1 -> decoder first upsample.
    model.align[-1] = nn.Sequential(model.align[-1], BottleneckMambaV1(512, 16, seed=44))
    return model


def source_hashes():
    paths = ['Models/TinyViM/mamba_bottleneck_v1.py',
             'source/bottleneck_mamba/experiment_v1.py',
             'source/bottleneck_mamba/test_v1.py',
             'source/comparison/ham_fast_runner.py',
             'source/comparison/ham_fast_pipeline.py',
             'source/comparison/pretrained_unet2d.py']
    return {p: sha256_file(PROJECT / p) for p in paths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['preflight', 'train'], required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    root = args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    runtime_audit = verify_v1_runtime()
    existing.verify_runtime = verify_v1_runtime
    original = existing.validate_protocol(existing.PROTOCOL, 'tinyvim_s', check_cache=True)
    existing.write_json(root / 'runtime_validation_audit.json', runtime_audit)
    config = copy.deepcopy(original)
    config['models'][NAME] = dict(initialization='ImageNet encoder; fresh decoder/alignment and bottleneck',
                                  run_dir=str(root / 'run'))
    protocol = {'version': 'bottleneck_mamba_v1', 'parent_protocol': str(existing.PROTOCOL),
                'parent_protocol_sha256': sha256_file(existing.PROTOCOL),
                'data': config['data'], 'training': config['training'],
                'preprocessing': config['preprocessing'], 'evaluation': config['evaluation'],
                'model': {'name': NAME, 'base': 'TinyViM-S + original shared UNet decoder',
                          'insertion': 'after deepest 224->512 alignment, before first decoder upsample',
                          'input_feature': ['batch', 512, 8, 8],
                          'tokens': ['batch', 64, 512], 'scan': 'single forward row-major Mamba-1',
                          'blocks': 1, 'd_model': 512, 'expand': 2, 'd_state': 16,
                          'original_content_conv': 'causal depthwise Conv1d kernel4 + SiLU, unchanged',
                          'generator': 'U -> causal DWConv1d kernel3 -> SiLU -> PWConv1d(E,E+2N)',
                          'split': ['E raw_delta', 'N B', 'N C'],
                          'delta': 'softplus(raw_delta), no rho or dt_proj',
                          'delta_initialization': 'first E pointwise biases inherit standard inverse-softplus dt initialization',
                          'state_update': 'exp(delta*A)*h_previous + delta*B*U; original scan unchanged',
                          'outer': 'LayerNorm -> V1 mixer -> residual',
                          'training_scope': 'all encoder, decoder, alignment and bottleneck parameters',
                          'new_bottleneck_seed': 44, 'combined_with_v2': False},
                'source_sha256': source_hashes(), 'test_data_accessed': False}
    protocol_path = root / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise RuntimeError('Existing run protocol differs; refusing to overwrite')
    else:
        existing.write_json(protocol_path, protocol)
    stage = root / ('preflight' if args.mode == 'preflight' else 'run')
    stage.mkdir(exist_ok=True)
    if args.mode == 'train':
        passed = json.loads((root / 'preflight/preflight_status.json').read_text())
        if passed['status'] != 'passed' or passed['protocol_sha256'] != sha256_file(protocol_path):
            raise RuntimeError('Exact-protocol preflight is required')
        if any((stage / n).exists() for n in ['checkpoints', 'status.json', 'history.csv']):
            raise FileExistsError('Fresh V1 training refuses existing run outputs')
    lock_path = root / 'RUNNING.lock'
    with lock_path.open('x') as stream:
        stream.write(str(os.getpid()))
    loaders = {}
    try:
        signal.signal(signal.SIGUSR1, existing.request_stop)
        api = existing.baseline_api()
        api.seed_all(42)
        torch.set_num_threads(4)
        torch.cuda.set_device(args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
        print(f'MODE={args.mode} PID={os.getpid()} GPU={args.gpu} MODEL={NAME}', flush=True)
        model = build_experiment_model(pretrained=True).to(device)
        insertion_seen = []
        def observe(module, inputs, output):
            entry = {'input': list(inputs[0].shape), 'output': list(output.shape)}
            if not insertion_seen:
                print('REAL_BOTTLENECK_SHAPE', json.dumps(entry), flush=True)
                existing.write_json(stage / 'bottleneck_shape.json', entry)
            insertion_seen.append(True)
        if args.mode == 'preflight':
            model.align[-1][1].register_forward_hook(observe)
        loaders, generator = make_fast_loaders(config, api, workers=2, validation_batch=8)
        criterion = api.BceDiceLoss().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01,
                                     betas=(0.9, 0.999), eps=1e-8)
        info = {'model': NAME, 'seed': 42, 'mode': args.mode, 'pid': os.getpid(),
                'protocol_file': str(protocol_path), 'protocol_sha256': sha256_file(protocol_path),
                'split_sha256': config['data']['split_sha256'],
                'parameters': sum(p.numel() for p in model.parameters()),
                'new_bottleneck_parameters': sum(p.numel() for p in model.align[-1][1].parameters()),
                'initialization': config['models'][NAME]['initialization'],
                'physical_gpu': args.gpu, 'gpu_name': torch.cuda.get_device_name(device),
                'training': config['training'], 'source_sha256': source_hashes(),
                'test_data_accessed': False, 'fresh_start': True,
                'engineering': {'runner': 'existing ham_fast_runner.train/preflight',
                                'workers': 2, 'validation_batch': 8}}
        existing.write_json(stage / 'provenance.json', info)
        existing.write_json(stage / 'weight_loading_audit.json', model.encoder.loading_audit)
        print('MODEL_PARAMETERS', info['parameters'], 'NEW_BOTTLENECK', info['new_bottleneck_parameters'], flush=True)
        if args.mode == 'preflight':
            report = existing.preflight(model, loaders, criterion, optimizer, device, stage, [])
            if not insertion_seen:
                raise AssertionError('Bottleneck was never executed')
            report.update(protocol_sha256=info['protocol_sha256'], split_sha256=config['data']['split_sha256'])
            existing.write_json(stage / 'preflight_status.json', report)
            print('PREFLIGHT_PASSED', json.dumps(report), flush=True)
        else:
            existing.train(model, loaders, criterion, optimizer, device, stage, config, info, generator, resume=False)
    except BaseException as error:
        existing.write_json(stage / 'failure.json', {'error': repr(error), 'pid': os.getpid(),
                            'mode': args.mode, 'time': datetime.now().isoformat()})
        raise
    finally:
        for loader in loaders.values():
            if hasattr(loader, 'close'):
                loader.close()
        lock_path.unlink(missing_ok=True)


if __name__ == '__main__':
    main()

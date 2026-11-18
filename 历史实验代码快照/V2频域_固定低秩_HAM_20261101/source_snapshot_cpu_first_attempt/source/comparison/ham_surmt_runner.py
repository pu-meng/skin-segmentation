"""Full SU-RMT comparison under the immutable HAM10000 V5 training protocol."""
import argparse
import json
import os
import signal
import subprocess
from pathlib import Path

import torch
from project_paths import PROJECT, CONFIGS, current_protocol

import ham_fast_runner as base
from surmt_adapter import SURMTV5

ROOT = Path(__file__).resolve().parent
PROTOCOL = CONFIGS / 'protocol_v5_surmt_extension_seed42.json'
DIAGNOSTICS = ROOT / 'diagnostics/surmt_extension_20261024'
MODEL = 'su_rmt'


def validate():
    config = base.validate_protocol(PROTOCOL, MODEL)
    parent = config['extension']['parent_protocol']
    if base.sha256_file(parent['path']) != parent['sha256']:
        raise ValueError('Parent V5 protocol changed')
    original = current_protocol(parent['path'])
    for key in ('data', 'training', 'preprocessing', 'evaluation', 'normalization'):
        if config[key] != original[key]:
            raise ValueError('Training/data/evaluation mismatch: ' + key)
    return config


def make_model(name, config):
    if name != MODEL:
        raise ValueError('Unexpected model')
    model = SURMTV5(pretrained=True)
    if sum(p.numel() for p in model.parameters()) != config['models'][MODEL]['parameters']:
        raise ValueError('SU-RMT parameter count changed')
    return model


def gpu_foreign_pids(gpu):
    uuid = subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-gpu=uuid',
                                    '--format=csv,noheader'], text=True).strip()
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                     '--format=csv,noheader,nounits'], text=True)
    pids = []
    for row in output.splitlines():
        parts = [p.strip() for p in row.split(',')]
        if len(parts) == 2 and parts[1] == uuid and int(parts[0]) != os.getpid():
            pids.append(int(parts[0]))
    return pids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['check', 'preflight', 'train'], required=True)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = validate()
    if args.mode == 'check':
        print(json.dumps({'status': 'passed', 'model': MODEL,
                          'protocol_sha256': base.sha256_file(PROTOCOL),
                          'counts': config['data']['counts'], 'training': config['training'],
                          'architecture': config['extension']['architecture_scope']}, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    if gpu_foreign_pids(args.gpu):
        raise RuntimeError('Selected GPU is occupied; use the waiting queue')
    signal.signal(signal.SIGUSR1, base.request_stop)
    api = base.baseline_api()
    api.seed_all(42)
    torch.set_num_threads(4)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}')
    formal = Path(config['models'][MODEL]['run_dir'])
    run_dir = formal if args.mode == 'train' else DIAGNOSTICS / 'gpu_preflight'
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = run_dir / 'RUNNING.lock'
    lock_owned = False
    status_file = run_dir / ('status.json' if args.mode == 'train' else 'preflight_status.json')
    try:
        if args.mode == 'train':
            pf = json.loads((DIAGNOSTICS / 'gpu_preflight/preflight_status.json').read_text())
            if pf['status'] != 'passed' or pf['protocol_sha256'] != base.sha256_file(PROTOCOL):
                raise RuntimeError('Exact-protocol GPU preflight required')
            if not args.resume and any((run_dir / n).exists() for n in
                    ('history.csv', 'checkpoints', 'status.json', 'RUNNING.lock')):
                raise FileExistsError('Fresh run refuses existing outputs')
        with lock.open('x') as stream:
            stream.write(str(os.getpid())+'\n')
        lock_owned = True
        model = make_model(MODEL, config).to(device)
        loaders, generator = base.make_fast_loaders(config, api, 2, 8)
        criterion = api.BceDiceLoss().to(device)
        t = config['training']
        optimizer = torch.optim.AdamW(model.parameters(), lr=t['lr'], weight_decay=t['weight_decay'],
                                     betas=tuple(t['betas']), eps=t['eps'])
        info = {'model': MODEL, 'seed': t['seed'], 'mode': args.mode, 'pid': os.getpid(),
                'protocol_file': str(PROTOCOL), 'protocol_sha256': base.sha256_file(PROTOCOL),
                'split_sha256': config['data']['split_sha256'], 'source_lock': config['source_lock'],
                'external_source_lock': config['external_source_lock'], 'training': t,
                'parameters': config['models'][MODEL]['parameters'],
                'initialization': config['models'][MODEL]['initialization'],
                'pretrained_source': config['models'][MODEL]['pretrained_source'],
                'physical_gpu': args.gpu, 'gpu_name': torch.cuda.get_device_name(device),
                'torch': torch.__version__, 'fresh_start': not args.resume, 'test_data_accessed': False,
                'decoder_initialization': 'Native SU-RMT decoder with official encoder-mirrored initialization',
                'engineering': {'runner': 'ham_fast_runner.py', 'workers': 2,
                    'validation_batch_size': 8, 'precise_validation_batch_size': 1, 'amp': False,
                    'architecture_scope': 'Full official SU-RMT; shared U-Net decoder is not substituted'}}
        base.write_json(run_dir / 'weight_loading_audit.json', model.loading_audit)
        base.write_json(run_dir / ('resume_provenance.json' if args.resume else 'provenance.json'), info)
        if args.mode == 'preflight':
            allowed = config['models'][MODEL]['allowed_missing_gradients']
            report = base.preflight(model, loaders, criterion, optimizer, device, run_dir, allowed)
            report.update(model=MODEL, protocol_sha256=info['protocol_sha256'],
                          split_sha256=info['split_sha256'], physical_gpu=args.gpu)
            base.write_json(status_file, report)
            print('SU-RMT GPU PREFLIGHT PASSED', flush=True)
        else:
            import recover_ham_dedup
            recover_ham_dedup.PROTOCOL = PROTOCOL
            base.train(model, loaders, criterion, optimizer, device, run_dir,
                       config, info, generator, resume=args.resume)
    except BaseException as error:
        if lock_owned:
            base.write_json(status_file, {'status': 'failed', 'model': MODEL, 'mode': args.mode,
                                         'pid': os.getpid(), 'error': repr(error)})
        raise
    finally:
        for loader in locals().get('loaders', {}).values():
            if hasattr(loader, 'close'):
                loader.close()
        if lock_owned:
            lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()

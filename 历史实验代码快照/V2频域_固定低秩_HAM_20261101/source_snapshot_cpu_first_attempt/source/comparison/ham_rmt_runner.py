"""RMT extension: reuse the locked V5 fast training/evaluation implementation."""
import argparse
import json
import os
import signal
from pathlib import Path

import torch
from project_paths import PROJECT, CONFIGS

import ham_fast_runner as base
from rmt_unet2d import RMTUNet2D, VARIANTS

ROOT = Path(__file__).resolve().parent
PROTOCOL = CONFIGS / 'protocol_v5_rmt_extension_seed42.json'
DIAGNOSTICS = ROOT / 'diagnostics/rmt_extension_20261023'


def make_model(name, config):
    model = RMTUNet2D(name, pretrained=True)
    count = sum(p.numel() for p in model.parameters())
    if count != config['models'][name]['parameters'] or count >= 50_000_000:
        raise ValueError('Unexpected RMT parameter count: ' + str(count))
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['check', 'preflight', 'train'], required=True)
    parser.add_argument('--model', choices=list(VARIANTS), required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = base.validate_protocol(PROTOCOL, args.model)
    parent = config['extension']['parent_protocol']
    if base.sha256_file(parent['path']) != parent['sha256']:
        raise ValueError('Parent V5 protocol changed')
    if args.mode == 'check':
        print(json.dumps({'status': 'passed', 'model': args.model,
                          'counts': config['data']['counts'], 'training': config['training']}, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in this terminal; no preflight or training started')
    signal.signal(signal.SIGUSR1, base.request_stop)
    api = base.baseline_api()
    api.seed_all(42)
    torch.set_num_threads(4)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}')
    formal = Path(config['models'][args.model]['run_dir'])
    run_dir = formal if args.mode == 'train' else DIAGNOSTICS / args.model
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = None
    lock_owned = False
    state = run_dir / ('status.json' if args.mode == 'train' else 'preflight_status.json')
    try:
        if args.mode == 'train':
            audit = json.loads((DIAGNOSTICS / args.model / 'preflight_status.json').read_text())
            if audit['status'] != 'passed' or audit['protocol_sha256'] != base.sha256_file(PROTOCOL):
                raise RuntimeError('Exact-protocol GPU preflight required')
            if not args.resume and any((run_dir / n).exists() for n in
                                       ['history.csv', 'checkpoints', 'status.json', 'RUNNING.lock']):
                raise FileExistsError('Fresh run refuses existing training outputs')
            lock = run_dir / 'RUNNING.lock'
            with lock.open('x') as stream:
                stream.write(str(os.getpid()) + '\n')
            lock_owned = True
        model = make_model(args.model, config).to(device)
        loaders, generator = base.make_fast_loaders(config, api, 2, 8)
        criterion = api.BceDiceLoss().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01,
                                     betas=(0.9, 0.999), eps=1e-8)
        info = {'model': args.model, 'seed': 42, 'mode': args.mode, 'pid': os.getpid(),
                'protocol_file': str(PROTOCOL), 'protocol_sha256': base.sha256_file(PROTOCOL),
                'split_sha256': config['data']['split_sha256'], 'source_lock': config['source_lock'],
                'external_source_lock': config['external_source_lock'],
                'parameters': config['models'][args.model]['parameters'],
                'initialization': config['models'][args.model]['initialization'],
                'pretrained_source': config['models'][args.model]['pretrained_source'],
                'physical_gpu': args.gpu, 'gpu_name': torch.cuda.get_device_name(device),
                'torch': torch.__version__, 'training': config['training'],
                'test_data_accessed': False, 'fresh_start': not args.resume,
                'decoder_initialization': 'SharedDecoder CPU seed42; alignment seed43',
                'engineering': {'runner': 'ham_fast_runner.py', 'workers': 2,
                                'validation_batch_size': 8, 'amp': False}}
        base.write_json(run_dir / 'weight_loading_audit.json', model.encoder.loading_audit)
        base.write_json(run_dir / ('resume_provenance.json' if args.resume else 'provenance.json'), info)
        if args.mode == 'preflight':
            report = base.preflight(model, loaders, criterion, optimizer, device, run_dir, [])
            report.update(model=args.model, protocol_sha256=info['protocol_sha256'],
                          split_sha256=config['data']['split_sha256'])
            base.write_json(state, report)
            print(args.model, 'GPU PREFLIGHT PASSED', flush=True)
        else:
            # The inherited resume validator reads this module global. This is
            # a process-local binding; the original file/protocol stays intact.
            import recover_ham_dedup
            recover_ham_dedup.PROTOCOL = PROTOCOL
            base.train(model, loaders, criterion, optimizer, device, run_dir,
                       config, info, generator, resume=args.resume)
    except BaseException as error:
        if args.mode == 'preflight' or lock_owned:
            base.write_json(state, {'status': 'failed', 'model': args.model, 'mode': args.mode,
                                    'pid': os.getpid(), 'error': repr(error)})
        raise
    finally:
        for loader in locals().get('loaders', {}).values():
            if hasattr(loader, 'close'):
                loader.close()
        if lock_owned:
            lock.unlink()


if __name__ == '__main__':
    main()

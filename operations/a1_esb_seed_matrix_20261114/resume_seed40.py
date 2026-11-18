#!/usr/bin/env python3
"""Resume the interrupted E+S+B ISIC seed40 run from its verified last state."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path('/home/PuMengYu/experiments/skin_segmentation')
OPS = ROOT / 'operations/a1_esb_seed_matrix_20261114'
OUT = Path('/media/8T/PuMengYu/experiments/skin_segmentation')
RUN_ROOT = OUT / 'A1_SWIFTFORMER_ESB_ISIC_SEED40_FRESH300_20261114'
LABEL = 'E+S+B'
ARM = 'E+S+B_bottle_only'
SEED = 40

sys.path.insert(0, str(ROOT))
from operations.a1_es_repeats_patience50_20261114 import entry


def recovery_inspect(model_name, config):
    """Audit this run's own protocol; the shared legacy inspector assumes HAM seed42."""
    import torch

    run = RUN_ROOT / 'ISIC2018' / ARM / 'run'
    checkpoint = run / 'checkpoints/latest.pth'
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    required = {
        'epoch', 'model', 'optimizer', 'scheduler', 'best_loss', 'best_epoch',
        'protocol_sha256', 'split_sha256', 'python_rng', 'numpy_rng', 'torch_rng',
        'cuda_rng', 'loader_rng',
    }
    if not required.issubset(payload):
        raise RuntimeError(f'checkpoint lacks resume state: {sorted(required - payload.keys())}')
    protocol = entry.read(RUN_ROOT / 'protocol.json')
    protocol_sha = entry.sha(RUN_ROOT / 'protocol.json')
    if payload['protocol_sha256'] != protocol_sha:
        raise RuntimeError('latest checkpoint does not match this run protocol')
    if payload['split_sha256'] != config['data']['split_sha256']:
        raise RuntimeError('latest checkpoint split hash differs from the frozen split')
    provenance = entry.read(run / 'provenance.json')
    if provenance['model'] != model_name or provenance['protocol_sha256'] != protocol_sha:
        raise RuntimeError('checkpoint provenance does not match this seed40 run')
    with (run / 'history.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    epoch = int(payload['epoch'])
    if [int(row['epoch']) for row in rows] != list(range(1, epoch + 1)):
        raise RuntimeError('history and latest checkpoint do not end at the same epoch')
    expected_updates = epoch * config['training']['updates_per_epoch']
    if int(rows[-1]['optimizer_updates']) != expected_updates:
        raise RuntimeError('optimizer-update count does not match the saved epoch')
    best = min(rows, key=lambda row: float(row['val_loss']))
    if int(best['epoch']) != int(payload['best_epoch']) or float(best['val_loss']) != float(payload['best_loss']):
        raise RuntimeError('best checkpoint metadata does not match validation history')
    if not (run / 'checkpoints/best.pth').is_file():
        raise RuntimeError('best checkpoint is missing')
    if (run / 'artifact_audit.json').exists():
        audit = entry.read(run / 'artifact_audit.json')
        if audit.get('training_complete'):
            raise RuntimeError('training is already marked complete; refusing duplicate resume')
    return {
        'action': 'resume', 'completed_epoch': epoch, 'next_epoch': epoch + 1,
        'optimizer_updates': expected_updates, 'best_epoch': int(payload['best_epoch']),
        'protocol_sha256': protocol_sha, 'split_sha256': payload['split_sha256'],
        'checkpoint': str(checkpoint),
    }


def main():
    entry.SEED = SEED
    entry.RUNS = {LABEL: RUN_ROOT}
    entry.ARMS = {LABEL: ARM}
    protocol = entry.verify(LABEL)
    torch, data, build, arm = entry.imports(LABEL)
    torch.set_num_threads(4)
    api = entry.runner.baseline_api()
    api.seed_all(SEED)
    model = build(arm, pretrained=True)
    work = RUN_ROOT / 'ISIC2018' / ARM
    if entry.digest_state(model.state_dict()) != entry.read(work / 'cpu_acceptance.json')['initial_state_sha256']:
        raise RuntimeError('seed40 initial model differs from the recorded CPU acceptance state')
    stage = work / 'run'
    optimizer_config = protocol['config']['training']
    torch.cuda.set_device(0)
    model.cuda()
    loaders, generator = data.loaders(protocol['config'], api)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=optimizer_config['lr'],
        weight_decay=optimizer_config['weight_decay'],
        betas=tuple(optimizer_config['betas']), eps=optimizer_config['eps'],
    )
    criterion = api.BceDiceLoss().cuda()

    # The shared runner's generic resume inspector is hard-wired to a different
    # HAM seed42 layout. Replace only that runtime audit hook with this run-specific
    # protocol/history/checkpoint verification; the training implementation stays frozen.
    import recover_ham_dedup
    recover_ham_dedup.inspect_run = recovery_inspect
    record = recovery_inspect(f'A1 {LABEL} ISIC seed{SEED}', protocol['config'])
    entry.write(OPS / 'resume_seed40_audit.json', record)
    info = entry.read(stage / 'provenance.json')
    info.update(pid=__import__('os').getpid(), recovery_from=record['checkpoint'], resumed=True)
    entry.write(OPS / 'resume_seed40_started.json', {
        'pid': info['pid'], **record, 'started_at': __import__('time').time(),
    })
    print('RECOVERY_AUDIT_PASS', json.dumps(record, ensure_ascii=False), flush=True)
    try:
        entry.runner.train(model, loaders, criterion, optimizer, torch.device('cuda:0'),
                           stage, protocol['config'], info, generator, resume=True)
        audit = entry.read(stage / 'artifact_audit.json')
        if not audit.get('training_complete') or not audit.get('validation_artifacts_complete'):
            raise RuntimeError(f'training/validation artifact audit incomplete: {audit}')
        if audit.get('epochs') != 300:
            raise RuntimeError(f'expected 300 epochs, got {audit.get("epochs")}')
    finally:
        for loader in loaders.values():
            if hasattr(loader, 'close'):
                loader.close()
            elif getattr(loader, '_iterator', None) is not None:
                loader._iterator._shutdown_workers()


if __name__ == '__main__':
    main()

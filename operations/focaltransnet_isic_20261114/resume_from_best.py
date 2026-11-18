#!/usr/bin/env python3
"""Continue the interrupted FocalTransNet run from its saved validation-best state."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/home/PuMengYu/experiments/skin_segmentation')
SOURCE_RUN = Path('/media/8T/PuMengYu/experiments/skin_segmentation/FocalTransNet_ISIC_FRESH100_GPU0_20261114')
RUN = Path('/media/8T/PuMengYu/experiments/skin_segmentation/FocalTransNet_ISIC_FRESH100_GPU0_RESUME_FROMBEST44_20261114')
ORIGINAL = ROOT / 'operations/focaltransnet_isic_20261114/run.py'
PHYSICAL_GPU = int(os.environ.get('EXPERIMENT_PHYSICAL_GPU', '0'))
GPU_UUID = os.environ.get('EXPERIMENT_GPU_UUID', 'GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde')
SPEC = importlib.util.spec_from_file_location('focaltransnet_original', ORIGINAL)
original = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(original)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def preflight(device: str) -> None:
    cfg = json.loads(original.CONFIG_PATH.read_text())
    original.seed_everything(cfg['author_seed'])
    payload = torch.load(SOURCE_RUN / 'best.pt', map_location='cpu', weights_only=False)
    target_device = torch.device(device)
    loader = original.make_loader('train', True, cfg)
    model = original.build_model().to(target_device)
    model.load_state_dict(payload['state_dict'], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5, betas=(0.9, 0.999), weight_decay=0.03)
    images, masks, _ = next(iter(loader))
    images = images.to(target_device, non_blocking=True)
    masks = masks.to(target_device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    loss, _, _ = original.OfficialIsicLoss()(logits, masks)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not torch.isfinite(loss) or not grads or not all(torch.isfinite(g).all() for g in grads):
        raise RuntimeError('FocalTransNet recovery preflight produced non-finite/missing gradients')
    optimizer.step()
    print(json.dumps({'preflight': 'passed', 'batch': len(images), 'device': device,
                      'physical_gpu': PHYSICAL_GPU, 'gpu_uuid': GPU_UUID,
                      'source_best_epoch': int(payload['epoch']),
                      'loss': float(loss.detach()), 'checkpoint_sha256': sha(SOURCE_RUN / 'best.pt')}),
          flush=True)


def main(device: str, resume: bool) -> None:
    source_best = SOURCE_RUN / 'best.pt'
    if not source_best.is_file():
        raise FileNotFoundError(source_best)
    cfg = json.loads(original.CONFIG_PATH.read_text())
    if resume:
        if not RUN.is_dir() or not (RUN / 'last.pt').is_file():
            raise FileNotFoundError('recovery --resume requires recovery run/last.pt')
        source = torch.load(RUN / 'last.pt', map_location='cpu', weights_only=False)
        start_epoch = int(source['epoch']) + 1
    else:
        if RUN.exists() and any(RUN.iterdir()):
            raise FileExistsError(f'refuse to overwrite recovery run: {RUN}')
        source = torch.load(source_best, map_location='cpu', weights_only=False)
        start_epoch = int(source['epoch']) + 1
        if start_epoch != 45:
            raise RuntimeError(f'expected saved best at epoch 44, got {start_epoch - 1}')
        RUN.mkdir(parents=True)
        (RUN / 'config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + '\n')
        import shutil
        shutil.copy2(SOURCE_RUN / 'history.csv', RUN / 'source_partial_history.csv')
        with (RUN / 'continuation_protocol.json').open('w') as f:
            json.dump({
                'source_run': str(SOURCE_RUN), 'source_best_checkpoint': str(source_best),
                'source_checkpoint_sha256': sha(source_best), 'source_best_epoch': 44,
                'continue_through_epoch': 100, 'gpu': PHYSICAL_GPU, 'gpu_uuid': GPU_UUID,
                'recovery_note': 'No last-state checkpoint existed. Continue from the saved best epoch 44 with optimizer/scheduler state; RNG state from the interrupted process was unavailable, so augmentation/order RNG restarts from configured seed.',
                'original_partial_history_preserved_as': 'source_partial_history.csv',
                'original_run_script': str(ORIGINAL), 'original_run_script_sha256': sha(ORIGINAL),
            }, f, ensure_ascii=False, indent=2)

    cfg_seed = cfg['author_seed']
    if not resume:
        original.seed_everything(cfg_seed)
    target_device = torch.device(device)
    train_loader = original.make_loader('train', True, cfg)
    val_loader = original.make_loader('val', False, cfg)
    model = original.build_model().to(target_device)
    model.load_state_dict(source['state_dict'], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5, betas=(0.9, 0.999), weight_decay=0.03)
    optimizer.load_state_dict(source['optimizer'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20000)
    scheduler.load_state_dict(source['scheduler'])
    if not resume:
        import shutil
        shutil.copy2(source_best, RUN / 'best.pt')
    else:
        if source.get('python_rng_state') is not None:
            random.setstate(source['python_rng_state'])
        if source.get('numpy_rng_state') is not None:
            np.random.set_state(source['numpy_rng_state'])
        if source.get('torch_rng_state') is not None:
            torch.set_rng_state(source['torch_rng_state'])
        if source.get('cuda_rng_state_all') is not None:
            torch.cuda.set_rng_state_all(source['cuda_rng_state_all'])
    criterion = original.OfficialIsicLoss()
    best = float(source['val_global_dice'])
    best_epoch = int(source.get('best_epoch', source['epoch']))
    history = RUN / 'history.csv'
    fields = ['epoch', 'train_loss', 'val_global_dice', 'lr', 'seconds']
    mode = 'a' if resume else 'w'
    with history.open(mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not resume:
            writer.writeheader(); f.flush()
        for epoch in range(start_epoch, 101):
            started = time.time()
            model.train(); losses = []
            for images, masks, _ in train_loader:
                images = images.to(target_device, non_blocking=True)
                masks = masks.to(target_device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss, _, _ = criterion(logits, masks)
                loss.backward(); optimizer.step(); scheduler.step()
                losses.append(float(loss.detach()))
            val = original.global_dice(model, val_loader, target_device)
            row = {'epoch': epoch, 'train_loss': float(np.mean(losses)),
                   'val_global_dice': val['global_dice'],
                   'lr': optimizer.param_groups[0]['lr'], 'seconds': time.time() - started}
            writer.writerow(row); f.flush()
            state = {'epoch': epoch, 'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                     'val_global_dice': best, 'best_epoch': best_epoch, 'config': cfg,
                     'python_rng_state': random.getstate(), 'numpy_rng_state': np.random.get_state(),
                     'torch_rng_state': torch.get_rng_state(),
                     'cuda_rng_state_all': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
            if val['global_dice'] > best:
                best = val['global_dice']; best_epoch = epoch
                state['val_global_dice'] = best
                state['best_epoch'] = best_epoch
                save(RUN / 'best.pt', state)
            save(RUN / 'last.pt', state)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    (RUN / 'train_complete.json').write_text(json.dumps({
        'epochs': 100, 'trained_epochs_this_stage': 56,
        'resumed_from_best_epoch': 44, 'best_epoch': best_epoch,
        'best_val_global_dice': best, 'resume_rng_was_restored': False,
    }, ensure_ascii=False, indent=2) + '\n')
    print('TRAINING_COMPLETE', str(RUN), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('device')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        preflight(args.device)
    else:
        main(args.device, args.resume)

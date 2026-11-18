"""Five-model HAM runner on the revised split; test evaluation stays deferred.

Engineering-optimized runner; frozen model, augmentation math, loss and protocol stay unchanged.
"""
import argparse
import csv
import hashlib
import importlib
import json
import os
import random
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from ham_fast_pipeline import make_fast_loaders, evaluate_fast

from pretrained_unet2d import (MATERIALS, PretrainedUNet2D, prepare_imports,
                                sha256_file)

from project_paths import (PROJECT, CONFIGS, RUNS, CACHE, current_protocol,
                           original_source, original_repository_source, verify_runtime)
ROOT = Path(__file__).resolve().parents[2] / "source/comparison"
from source.recipe_r1_20261105.recipe import UpdateSchedule,clip
PROTOCOL = CONFIGS / 'protocol_v5_dedup_seed42.json'
STOP_REQUESTED = False

def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True




def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def baseline_api():
    # Use real installed Mamba modules without importing an unrelated language LM.
    prepare_imports('mambavision_t')
    return importlib.import_module('baseline_runner_seed42')


def validate_protocol(protocol_path, model_name, check_cache=True):
    verify_runtime()
    config = current_protocol(protocol_path)
    if config['version'] != 'ham_dedup_v5_seed42_20261023':
        raise ValueError('Wrong protocol version')
    if sha256_file(protocol_path) != protocol_path.with_suffix('.sha256').read_text().strip():
        raise ValueError('Protocol SHA256 mismatch')
    if model_name not in config['models']:
        raise ValueError('Model is outside the fixed three-encoder plus two-baseline scope')
    for relative, expected in config['source_lock'].items():
        if sha256_file(original_repository_source(relative)) != expected:
            raise ValueError('Frozen source changed: ' + relative)
    for absolute, expected in config['external_source_lock'].items():
        if sha256_file(original_source(absolute)) != expected:
            raise ValueError('Frozen official model source changed: ' + absolute)
    split = Path(config['data']['split_file'])
    if sha256_file(split) != config['data']['split_sha256']:
        raise ValueError('Revised split hash mismatch')
    if sha256_file(split.parent / 'split_audit.json') != config['data']['split_audit_sha256']:
        raise ValueError('Split audit hash mismatch')
    for filename, expected in config['data']['exclusion_artifact_hashes'].items():
        if sha256_file(split.parent / filename) != expected:
            raise ValueError('Exclusion decision/manifest changed: ' + filename)
    with split.open() as stream:
        rows = list(csv.DictReader(stream))
    counts = {s: sum(r['split'] == s for r in rows) for s in ('train', 'val', 'test')}
    if counts != config['data']['counts']:
        raise ValueError('Split counts differ from the protocol')
    if len({r['image_id'] for r in rows}) != len(rows):
        raise ValueError('Repeated image_id')
    active_ids = {r['image_id'] for r in rows}
    policy = json.loads((split.parent / 'exclusion_policy.json').read_text())
    for excluded, kept in policy['excluded_to_kept'].items():
        if excluded in active_ids or kept not in active_ids:
            raise ValueError('Approved photographic-variant exclusion not applied')
    for key in ('lesion_id', 'dedup_group_id'):
        sets = {s: {r[key] for r in rows if r['split'] == s} for s in counts}
        for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
            if sets[a] & sets[b]:
                raise ValueError(f'Cross-split {key}: {a}, {b}')
    if check_cache:
        for name, record in config['data']['cache_files'].items():
            path = CACHE / name
            if path.resolve() != Path(record['real_path']).resolve() or sha256_file(path) != record['sha256']:
                raise ValueError('Cache identity/content changed: ' + name)
    for name, record in config['models'].items():
        if record['initialization'] != 'random':
            source = record['pretrained_source']
            if sha256_file(source['weights_path']) != source['sha256']:
                raise ValueError('Pretrained weights changed: ' + name)
    if config['training']['epochs'] != 300 or config['training']['batch_size'] != 8:
        raise ValueError('Fixed training budget changed')
    if (counts['train'] + 7) // 8 != config['training']['updates_per_epoch']:
        raise ValueError('Incorrect update budget')
    return config


def make_model(name, config):
    api = baseline_api()
    if name == 'unet2d':
        model = api.UNet2D()
    elif name == 'mambalite':
        model = api.MambaLiteUNet()
    else:
        model = PretrainedUNet2D(name, pretrained=True, new_layers_seed=42)
    count = sum(p.numel() for p in model.parameters())
    if count != config['models'][name]['parameters'] or count >= 50_000_000:
        raise ValueError(f'Unexpected parameter count: {count}')
    return model


def make_loaders(config):
    api = baseline_api()
    # Retain the old baseline sampler stream: reference Mamba constructor at
    # seed42, isolated from the actual model's parameter initialization.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        reference = api.MambaLiteUNet()
        loader_state = torch.get_rng_state()
        del reference
    generator = torch.Generator()
    generator.set_state(loader_state)
    data_path = Path(config['data']['split_file']).parent
    loaders = {}
    # Do not even construct a test loader in this screening phase.
    for split in ('train', 'val'):
        dataset = api.HAMDataset(data_path, split)
        sampler = api.ReferenceRandomSampler(dataset, generator) if split == 'train' else None
        loaders[split] = DataLoader(dataset, batch_size=8 if split == 'train' else 1,
                                    sampler=sampler, num_workers=0, pin_memory=True,
                                    drop_last=False, generator=generator)
    if len(loaders['train']) != config['training']['updates_per_epoch']:
        raise ValueError('Loader steps differ from protocol')
    return loaders, generator


def preflight(model, loaders, criterion, optimizer, device, run_dir, allowed_missing):
    model.train()
    steps = []
    schedule = UpdateSchedule(optimizer)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    iterator = iter(loaders['train'])
    for index in range(3):
        x, y, ids, _ = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x.to(device))
        loss = criterion(prediction, y.to(device))
        if not torch.isfinite(loss) or tuple(prediction.shape) != (8, 1, 256, 256):
            raise RuntimeError('Invalid preflight loss/output')
        loss.backward()
        missing = {n for n, p in model.named_parameters() if p.requires_grad and p.grad is None}
        if missing != set(allowed_missing):
            raise RuntimeError(f'Unexpected missing-gradient set: {sorted(missing)}')
        norm = clip(model.parameters())
        schedule.apply()
        optimizer.step()
        schedule.advance()
        steps.append({'step': index + 1, 'loss': float(loss.detach()), 'image_ids': list(ids), 'preclip_norm': float(norm.detach()), 'group_lrs': [g['lr'] for g in optimizer.param_groups]})
    # Exercise actual revised validation data, using a tiny diagnostic subset.
    api = baseline_api()
    small_val = DataLoader(Subset(loaders['val'].dataset, [0, 1]), batch_size=1, num_workers=0)
    validation = api.evaluate(model, small_val, criterion, device)
    model.eval()
    batch = next(iter(small_val))[0].to(device)
    with torch.no_grad():
        expected = model(batch).detach().cpu()
    checkpoint = run_dir / 'temporary_preflight.pth'
    try:
        torch.save(model.state_dict(), checkpoint)
        model.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
        with torch.no_grad():
            actual = model(batch).detach().cpu()
        torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-6)
    finally:
        checkpoint.unlink(missing_ok=True)
    torch.cuda.synchronize(device)
    return {'status': 'passed', 'optimizer_steps': len(steps), 'steps': steps,
            'validation_diagnostic_images': validation['images'], 'validation_diagnostic_loss': validation['loss'],
            'checkpoint_roundtrip': True, 'preflight_weights_discarded': True,
            'formal_training_started': False, 'test_data_accessed': False,
            'inherited_unused_parameters': sorted(allowed_missing),
            'peak_allocated_mib': torch.cuda.max_memory_allocated(device) / 2**20,
            'peak_reserved_mib': torch.cuda.max_memory_reserved(device) / 2**20,
            'elapsed_seconds': time.monotonic() - started}


def train(model, loaders, criterion, optimizer, device, run_dir, config, provenance, generator, resume=False):
    api = baseline_api()
    scheduler = UpdateSchedule(optimizer)
    checkpoints = run_dir / 'checkpoints'
    checkpoints.mkdir(exist_ok=resume)
    history = run_dir / 'history.csv'
    best_loss, best_epoch = float('inf'), 0
    epochs = config['training']['epochs']
    state = {'status': 'training', 'pid': os.getpid(), 'model': provenance['model'],
             'epoch_completed': 0, 'optimizer_updates': 0, 'test_data_accessed': False}
    start_epoch = 1
    if resume:
        record = {'recovery':'R1 latest full-state epoch checkpoint'}
        payload = torch.load(checkpoints / 'latest.pth', map_location='cpu')
        model.load_state_dict(payload['model'], strict=True)
        optimizer.load_state_dict(payload['optimizer'])
        scheduler.load_state_dict(payload['scheduler'])
        assert scheduler.update==payload['global_update']==payload['epoch']*config['training']['updates_per_epoch']
        best_loss, best_epoch = payload['best_loss'], payload['best_epoch']
        start_epoch = payload['epoch'] + 1
        random.setstate(payload['python_rng'])
        np.random.set_state(payload['numpy_rng'])
        torch.set_rng_state(payload['torch_rng'])
        torch.cuda.set_rng_state(payload['cuda_rng'], device)
        generator.set_state(payload['loader_rng'])
        state.update(epoch_completed=payload['epoch'],
                     optimizer_updates=payload['epoch'] * config['training']['updates_per_epoch'])
        write_json(run_dir / 'resume_loaded.json', dict(record, pid=os.getpid(),
                   restored=['model', 'optimizer', 'scheduler', 'python_rng', 'numpy_rng',
                             'torch_rng', 'cuda_rng', 'loader_rng']))
        print(f'RESUMED completed_epoch={payload["epoch"]} next_epoch={start_epoch}', flush=True)
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        started, losses = time.monotonic(), []
        norms = []
        lr_used = optimizer.param_groups[0]['lr']
        state.update(epoch=epoch, status='training', updated_at=datetime.now().isoformat())
        write_json(run_dir / 'status.json', state)
        for step, (x, y, _, _) in enumerate(loaders['train'], 1):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x.to(device, non_blocking=True))
            loss = criterion(prediction, y.to(device, non_blocking=True))
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss before backward')
            loss.backward()
            norm = clip(model.parameters())
            norms.append(norm.detach())
            scheduler.apply()
            optimizer.step()
            scheduler.advance()
            losses.append(loss.detach())
            state['optimizer_updates'] += 1
            if step == 1 or step % 50 == 0 or step == len(loaders['train']):
                loss_values = torch.stack(losses).cpu().double().numpy()
                if not np.isfinite(loss_values).all():
                    raise RuntimeError(f'Nonfinite loss before checkpoint: epoch {epoch}, step {step}')
                state.update(step=step, steps_per_epoch=len(loaders['train']), train_loss=float(np.mean(loss_values)))
                write_json(run_dir / 'status.json', state)
                print(f'TRAIN epoch={epoch}/{epochs} step={step}/{len(loaders["train"])} loss={np.mean(loss_values):.6f} lr={lr_used:.8f}', flush=True)
        ns = torch.stack(norms)
        from source.recipe_r1_20261105.recipe import SPEC
        norm_stats = torch.stack([ns.mean(), ns.max(), (ns>SPEC['max_norm']).sum().to(ns.dtype)]).cpu().tolist()
        validation = api.evaluate(model, loaders['val_precise'], criterion, device)
        if validation['loss'] < best_loss:
            best_loss, best_epoch = validation['loss'], epoch
            torch.save(model.state_dict(), checkpoints / 'best.pth')
        payload = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                   'scheduler': scheduler.state_dict(), 'global_update':scheduler.update, 'best_loss': best_loss, 'best_epoch': best_epoch,
                   'protocol_sha256': provenance['protocol_sha256'], 'split_sha256': config['data']['split_sha256'],
                   'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                   'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(device),
                   'loader_rng': generator.get_state(),
                   'engineering': provenance.get('engineering')}
        temporary = checkpoints / 'latest.tmp.pth'
        torch.save(payload, temporary)
        temporary.replace(checkpoints / 'latest.pth')
        row = {'epoch': epoch, 'train_loss': float(np.mean(loss_values)), 'val_loss': validation['loss'],
               'val_global_dice': validation['global']['dice'], 'val_global_iou': validation['global']['iou'],
               'val_mean_dice': validation['per_image_mean']['dice'], 'val_mean_iou': validation['per_image_mean']['iou'],
               'lr': optimizer.param_groups[0]['lr'], 'other_lr':optimizer.param_groups[1]['lr'], 'preclip_norm_mean':norm_stats[0], 'preclip_norm_max':norm_stats[1], 'clip_trigger_count':int(norm_stats[2]), 'seconds': time.monotonic() - started, 'best_epoch': best_epoch,
               'optimizer_updates': state['optimizer_updates']}
        with history.open('a', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if epoch == 1:
                writer.writeheader()
            writer.writerow(row)
        state.update(epoch_completed=epoch, best_epoch=best_epoch, best_val_loss=best_loss,
                     validation=validation, epoch_seconds=row['seconds'])
        write_json(run_dir / 'status.json', state)
        print('EPOCH COMPLETE', json.dumps(row), flush=True)
        if STOP_REQUESTED and epoch < epochs:
            state['status'] = 'stopped_at_epoch_boundary'
            write_json(run_dir / 'status.json', state)
            print('STOPPED AT SAVED EPOCH', epoch, flush=True)
            return
    if state['optimizer_updates'] != config['training']['total_optimizer_updates']:
        raise RuntimeError('Optimizer update count differs from budget')
    torch.save(model.state_dict(), checkpoints / 'final.pth')
    model.load_state_dict(torch.load(checkpoints / 'best.pth', map_location=device), strict=True)
    final_loader = loaders['val_precise']
    result = api.evaluate(model, final_loader, criterion, device, run_dir / 'validation')
    result.update(dataset='HAM10000', split='val', selected_epoch=best_epoch,
                  checkpoint=str(checkpoints / 'best.pth'), checkpoint_sha256=sha256_file(checkpoints / 'best.pth'),
                  model=provenance['model'], seed=42, protocol_sha256=provenance['protocol_sha256'],
                  split_sha256=config['data']['split_sha256'], test_data_accessed=False)
    write_json(run_dir / 'validation/summary.json', result)
    (run_dir / 'validation/report.txt').write_text(
        f'HAM10000修订划分，{provenance["model"]}，seed42\n'
        f'{epochs}轮，{state["optimizer_updates"]}次更新；按验证损失选择epoch {best_epoch}\n'
        f'验证图像{result["images"]}张；全局Dice {100*result["global"]["dice"]:.4f}%，IoU {100*result["global"]["iou"]:.4f}%\n'
        '当前为验证结果，测试评估留待方法固定后执行。\n')
    expected_ids = {r['image_id'] + '_segmentation.png' for r in loaders['val'].dataset.rows}
    actual_ids = {p.name for p in (run_dir / 'validation/predictions').glob('*.png')}
    if expected_ids != actual_ids:
        raise RuntimeError('Validation prediction file set mismatch')
    with history.open() as stream:
        rows = list(csv.DictReader(stream))
    if [int(r['epoch']) for r in rows] != list(range(1, epochs + 1)):
        raise RuntimeError('History is incomplete')
    with (run_dir / 'validation/per_image_metrics.csv').open() as stream:
        per_image = list(csv.DictReader(stream))
    if {r['image_id'] for r in per_image} != {r['image_id'] for r in loaders['val'].dataset.rows}:
        raise RuntimeError('Validation per-image metrics mismatch')
    visuals = len(list((run_dir / 'validation/visualizations').glob('*.png')))
    if visuals != 12 or any(not (checkpoints / n).exists() for n in ['best.pth', 'latest.pth', 'final.pth']):
        raise RuntimeError('Checkpoint or visualization missing')
    artifact = {'training_complete': True, 'validation_artifacts_complete': True,
                'epochs': epochs, 'optimizer_updates': state['optimizer_updates'],
                'validation_predictions': len(actual_ids), 'per_image_metrics': len(per_image),
                'visualizations': visuals, 'selected_epoch': best_epoch,
                'test_evaluation_complete': False, 'test_evaluation_deferred': True}
    write_json(run_dir / 'artifact_audit.json', artifact)
    state.update(status='training_and_validation_complete_test_deferred', validation_summary=str(run_dir / 'validation/summary.json'))
    write_json(run_dir / 'status.json', state)


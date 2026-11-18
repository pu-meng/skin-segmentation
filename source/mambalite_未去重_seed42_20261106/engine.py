import csv,json,os,random,time,hashlib,importlib,importlib.util,importlib.metadata,sys,types
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from ham_fast_pipeline import make_fast_loaders,evaluate_fast
STOP_REQUESTED=False
def sha256_file(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def write_json(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def baseline_api():
 prepare_imports('mambavision_t')
 return importlib.import_module('baseline_runner_seed42')

def prepare_imports(variant):
    if variant == 'mambavision_t' and 'mamba_ssm' not in sys.modules:
        # mamba_ssm.__init__ imports an unrelated language model that requires old
        # transformers. Load the real installed ops submodule without that export.
        spec = importlib.util.find_spec('mamba_ssm')
        if spec is None:
            raise ImportError('Installed mamba_ssm is required')
        package = types.ModuleType('mamba_ssm')
        package.__path__ = list(spec.submodule_search_locations)
        package.__spec__ = spec
        package.__package__ = 'mamba_ssm'
        package.__version__ = importlib.metadata.version('mamba-ssm')
        sys.modules['mamba_ssm'] = package
        # Preserve real vision/baseline exports, omitting only language-model
        # imports. Neither the operators nor their autograd are replaced.
        from mamba_ssm.modules.mamba_simple import Mamba
        from mamba_ssm.modules.mamba2 import Mamba2
        package.Mamba, package.Mamba2 = Mamba, Mamba2

def preflight(model, loaders, criterion, optimizer, device, run_dir, allowed_missing):
    model.train()
    steps = []
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
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or any(not torch.isfinite(g).all() for g in gradients):
            raise RuntimeError('Nonfinite preflight gradient')
        optimizer.step()
        steps.append({'step': index + 1, 'loss': float(loss.detach()), 'image_ids': list(ids)})
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
    checkpoints = run_dir / 'checkpoints'
    checkpoints.mkdir(exist_ok=resume)
    history = run_dir / 'history.csv'
    best_loss, best_epoch = float('inf'), 0
    epochs = config['training']['epochs']
    state = {'status': 'training', 'pid': os.getpid(), 'model': provenance['model'],
             'epoch_completed': 0, 'optimizer_updates': 0, 'test_data_accessed': False}
    start_epoch = 1
    if resume:
        from recover_ham_dedup import inspect_run
        record = inspect_run(provenance['model'], config)
        payload = torch.load(checkpoints / 'latest.pth', map_location='cpu')
        model.load_state_dict(payload['model'], strict=True)
        optimizer.load_state_dict(payload['optimizer'])
        scheduler.load_state_dict(payload['scheduler'])
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
        lr_used = optimizer.param_groups[0]['lr']
        state.update(epoch=epoch, status='training', updated_at=datetime.now().isoformat())
        write_json(run_dir / 'status.json', state)
        for step, (x, y, _, _) in enumerate(loaders['train'], 1):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x.to(device, non_blocking=True))
            loss = criterion(prediction, y.to(device, non_blocking=True))
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
            state['optimizer_updates'] += 1
            if step == 1 or step % 50 == 0 or step == len(loaders['train']):
                loss_values = torch.stack(losses).cpu().double().numpy()
                if not np.isfinite(loss_values).all():
                    raise RuntimeError(f'Nonfinite loss before checkpoint: epoch {epoch}, step {step}')
                state.update(step=step, steps_per_epoch=len(loaders['train']), train_loss=float(np.mean(loss_values)))
                write_json(run_dir / 'status.json', state)
                print(f'TRAIN epoch={epoch}/{epochs} step={step}/{len(loaders["train"])} loss={np.mean(loss_values):.6f} lr={lr_used:.8f}', flush=True)
        scheduler.step()
        validation = evaluate_fast(model, loaders['val'], criterion, device, api)
        # Batch-shape rounding must not decide a new best checkpoint. Recheck
        # all potential improvements and near ties using the original evaluator.
        if validation['loss'] <= best_loss + 0.001:
            validation = api.evaluate(model, loaders['val_precise'], criterion, device)
        if validation['loss'] < best_loss:
            best_loss, best_epoch = validation['loss'], epoch
            torch.save(model.state_dict(), checkpoints / 'best.pth')
        payload = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                   'scheduler': scheduler.state_dict(), 'best_loss': best_loss, 'best_epoch': best_epoch,
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
               'lr': lr_used, 'seconds': time.monotonic() - started, 'best_epoch': best_epoch,
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

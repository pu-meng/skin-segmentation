"""HAM10000 runner around the unmodified public model and loss."""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset, Sampler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from models.MambaLiteUNet import MambaLiteUNet
from utils import BceDiceLoss
from unet2d import UNet2D


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class HAMDataset(Dataset):
    def __init__(self, data, split):
        self.images = np.load(data / 'images.npy', mmap_mode='r')
        self.masks = np.load(data / 'masks.npy', mmap_mode='r')
        with (data / 'split.csv').open() as stream:
            self.rows = [r for r in csv.DictReader(stream) if r['split'] == split]
        self.training = split == 'train'

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        i = int(row['cache_index'])
        image = self.images[i].astype(np.float64) / 255.0
        mask = (self.masks[i] > 0).astype(np.float32)
        if self.training:
            if random.random() > 0.5:
                k = np.random.randint(0, 4)
                axis = np.random.randint(0, 2)
                image = np.flip(np.rot90(image, k), axis=axis).copy()
                mask = np.flip(np.rot90(mask, k), axis=axis).copy()
            if random.random() > 0.5:
                angle = np.random.randint(20, 80)
                image = ndimage.rotate(image, angle, order=0, reshape=False)
                mask = ndimage.rotate(mask, angle, order=0, reshape=False)
        x = torch.tensor(image.copy(), dtype=torch.float32).permute(2, 0, 1)
        y = torch.tensor(mask.copy(), dtype=torch.float32).unsqueeze(0)
        return x, y, row['image_id'], row['dx']


def confusion(prediction, target):
    p, g = prediction.astype(bool), target.astype(bool)
    return np.array([(p & g).sum(), (p & ~g).sum(), (~p & g).sum(), (~p & ~g).sum()], dtype=np.int64)


def metrics(counts):
    tp, fp, fn, tn = map(int, counts)
    def ratio(a, b):
        return a / b if b else 0.0
    return {'dice': ratio(2 * tp, 2 * tp + fp + fn), 'iou': ratio(tp, tp + fp + fn),
            'accuracy': ratio(tp + tn, tp + fp + fn + tn), 'sensitivity': ratio(tp, tp + fn),
            'specificity': ratio(tn, tn + fp), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def hd95(prediction, target):
    p, g = prediction.astype(bool), target.astype(bool)
    if not p.any() or not g.any():
        return 0.0 if not p.any() and not g.any() else float(math.hypot(*p.shape))
    surface_p = p ^ ndimage.binary_erosion(p)
    surface_g = g ^ ndimage.binary_erosion(g)
    distances = np.concatenate((ndimage.distance_transform_edt(~surface_g)[surface_p],
                                ndimage.distance_transform_edt(~surface_p)[surface_g]))
    return float(np.percentile(distances, 95, method="linear"))


class ReferenceRandomSampler(Sampler):
    """Mirror RandomSampler(generator=None)'s local seed using an isolated RNG."""
    def __init__(self, dataset, generator):
        self.dataset, self.generator = dataset, generator

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        local_seed = int(torch.empty((), dtype=torch.int64).random_(generator=self.generator).item())
        local = torch.Generator().manual_seed(local_seed)
        return iter(torch.randperm(len(self.dataset), generator=local).tolist())


def loaders(data, batch, generator=None):
    result = {}
    for split in ('train', 'val', 'test'):
        dataset = HAMDataset(data, split)
        sampler = ReferenceRandomSampler(dataset, generator) if split == 'train' else None
        result[split] = DataLoader(dataset, batch_size=batch if split == 'train' else 1,
                                  sampler=sampler, num_workers=0, pin_memory=True,
                                  drop_last=False, generator=generator)
    return result


def evaluate(model, loader, criterion, device, output=None):
    model.eval()
    total = np.zeros(4, dtype=np.int64)
    categories = defaultdict(lambda: np.zeros(4, dtype=np.int64))
    records = []
    losses = []
    if output:
        (output / 'predictions').mkdir(parents=True, exist_ok=True)
        (output / 'visualizations').mkdir(exist_ok=True)
    with torch.no_grad():
        for index, (x, y, ids, classes) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            probability = model(x)
            losses.append(float(criterion(probability, y)))
            prediction = probability[0, 0].cpu().numpy() >= 0.5
            target = y[0, 0].cpu().numpy() >= 0.5
            count = confusion(prediction, target)
            total += count
            categories[classes[0]] += count
            record = {'image_id': ids[0], 'dx': classes[0], **metrics(count)}
            if output:
                record['prediction_empty'] = bool(not prediction.any())
                record['target_empty'] = bool(not target.any())
                record['hd95_pixels'] = hd95(prediction, target)
                Image.fromarray(prediction.astype(np.uint8) * 255).save(output / 'predictions' / f'{ids[0]}_segmentation.png')
                if index < 12:
                    rgb = np.clip(x[0].cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
                    gt = np.repeat(target[..., None].astype(np.uint8) * 255, 3, axis=2)
                    pr = np.repeat(prediction[..., None].astype(np.uint8) * 255, 3, axis=2)
                    Image.fromarray(np.concatenate((rgb, gt, pr), axis=1)).save(output / 'visualizations' / f'{ids[0]}_image_gt_prediction.png')
            records.append(record)
    result = {'loss': float(np.mean(losses)), 'global': metrics(total), 'images': len(records),
              'per_image_mean': {k: float(np.mean([r[k] for r in records])) for k in ('dice', 'iou')},
              'by_category_global': {k: metrics(v) for k, v in sorted(categories.items())}}
    if output:
        result['per_image_mean']['hd95_pixels'] = float(np.mean([r['hd95_pixels'] for r in records]))
        result['hd95_convention'] = 'pooled bidirectional surface distances; percentile95 linear; 4-neighbor boundary; spacing1 pixel; image mean; one-empty penalty sqrt(H^2+W^2)'
        result['empty_mask_counts'] = {
            'prediction_empty': sum(r['prediction_empty'] for r in records),
            'target_empty': sum(r['target_empty'] for r in records),
            'both_empty': sum(r['prediction_empty'] and r['target_empty'] for r in records),
            'prediction_empty_fraction': sum(r['prediction_empty'] for r in records) / len(records),
        }
        with (output / 'per_image_metrics.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    return result


def provenance(args, model):
    from importlib.metadata import version
    return {
        'git_commit': subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
        'model': args.model,
        'model_sha256': hashlib.sha256((Path(__file__).parent / 'unet2d.py' if args.model == 'unet2d' else REPO / 'models/MambaLiteUNet.py').read_bytes()).hexdigest(),
        'protocol_sha256': hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        'protocol_file': str(args.protocol),
        'sampler': 'shared dedicated generator initialized from this run seed public Mamba initialization RNG state; shared train/val/test loader stream',
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'split_sha256': hashlib.sha256((args.data / 'split.csv').read_bytes()).hexdigest(),
        'parameters': sum(p.numel() for p in model.parameters()),
        'environment': {n: version(n) for n in ('torch', 'torchvision', 'mamba-ssm', 'causal-conv1d', 'timm', 'transformers', 'numpy', 'scipy', 'pillow')},
        'python': sys.version, 'physical_gpu': args.gpu, 'gpu_name': torch.cuda.get_device_name(args.gpu),
        'protocol': {'epochs': args.epochs, 'batch_size': args.batch_size, 'seed': args.seed, 'size': 256,
                     'optimizer': 'AdamW', 'lr': 0.001, 'weight_decay': 0.01, 'betas': [0.9, 0.999],
                     'scheduler': 'CosineAnnealingLR', 'T_max': 50, 'eta_min': 1e-5,
                     'loss': 'unmodified official BceDiceLoss', 'amp': False, 'threshold': 0.5,
                     'selection': 'lowest validation loss', 'tta': False, 'source_weights': 'random initialization'},
        'limits': ['Exact author HAM split not recovered; user approved lesion-disjoint grouped split. MatchSeg candidate file exists but was not obtained.',
                   'Public Mamba has 910627 parameters versus paper 0.494M; UNet is explicitly defined base32 with BN.',
                   'Public Mamba differs from paper SS2D description; the basic UNet is our controlled baseline definition.',
                   'This run is one of five paired seeds; report five-run statistics only after all five are complete.',
                   'Runtime differences are recorded for provenance and are not an alignment requirement.',
                   'T_max=50 retained from public config; not silently changed to 300.'],
    }


def validate_formal_run(args, protocol):
    """Reject incompatible budgets, seeds, source files and existing formal outputs."""
    if protocol['version'] != 'ham_baseline_v2_20261021_final':
        raise ValueError('Formal runs require the confirmed protocol_v2.json')
    training = protocol['training']
    if args.epochs != training['epochs'] or args.batch_size != training['batch_size']:
        raise ValueError('Training budget differs from the frozen protocol')
    if args.seed not in training['seeds']:
        raise ValueError('Seed is outside the five confirmed seeds')
    if args.model not in protocol['active_scope']['allowed_models']:
        raise ValueError('Model is outside step-1 scope')
    if args.data.resolve() != Path(protocol['data']['split_file']).parent.resolve():
        raise ValueError('Data cache is not the frozen HAM cache')
    if hashlib.sha256((args.data / 'split.csv').read_bytes()).hexdigest() != protocol['data']['split_sha256']:
        raise ValueError('Data membership differs from the frozen split')
    for relative, expected in protocol['source_lock'].items():
        if hashlib.sha256((REPO / relative).read_bytes()).hexdigest() != expected:
            raise ValueError('Source differs from the frozen protocol: ' + relative)
    if args.mode == 'train':
        if (args.run_dir / 'history.csv').exists() or any((args.run_dir / 'checkpoints').glob('*.pth')):
            raise FileExistsError('From-scratch training refuses existing training history/checkpoints; use a new run directory')


def main():
    from pathlib import Path as _StopPath
    if (_StopPath(__file__).resolve().parent / 'EXPERIMENTS_STOPPED.json').exists():
        raise SystemExit('Experiments stopped by user: audit paper protocols before relaunch.')
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('preflight', 'train'), required=True)
    parser.add_argument('--model', choices=('unet2d', 'mambalite'), required=True)
    parser.add_argument('--protocol', type=Path, default=Path(__file__).parent / 'protocol_v2.json')
    parser.add_argument('--memory-fraction', type=float, default=0.35)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    validate_formal_run(args, protocol)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    state_file = args.run_dir / ('preflight_status.json' if args.mode == 'preflight' else 'status.json')
    state = {'status': 'initializing', 'mode': args.mode, 'pid': os.getpid(), 'gpu': args.gpu, 'epoch': 0}
    write_json(state_file, state)
    try:
        torch.set_num_threads(4)
        torch.cuda.set_device(args.gpu)
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
        seed_all(args.seed)
        # Reproduce the existing Mamba run's data RNG stream while decoupling it
        # from the candidate model's parameter initialization.
        with torch.random.fork_rng(devices=[]):
            reference = MambaLiteUNet()
            loader_state = torch.get_rng_state()
            del reference
        model = (UNet2D() if args.model == 'unet2d' else MambaLiteUNet()).to(device)
        generator = torch.Generator()
        generator.set_state(loader_state)
        data = loaders(args.data, args.batch_size, generator=generator)
        assert len(data['train']) == protocol['training']['updates_per_epoch'] == 874
        assert len(data['train'].dataset) == 6985
        assert sum(p.numel() for p in model.parameters()) < 50_000_000
        criterion = BceDiceLoss().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
        info = provenance(args, model)
        write_json(args.run_dir / 'provenance.json', info)
        print(json.dumps(info, indent=2), flush=True)
        if args.mode == 'preflight':
            model.train()
            steps = []
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats(device)
            for index, (x, y, ids, _) in enumerate(data['train']):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x)
                loss = criterion(prediction, y)
                assert torch.isfinite(loss)
                loss.backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                assert grads and all(torch.isfinite(g).all().item() for g in grads)
                nonzero = sum(int(torch.count_nonzero(g).item() > 0) for g in grads)
                assert nonzero > 0
                optimizer.step()
                torch.cuda.synchronize(device)
                steps.append({'loss': float(loss), 'shape': list(prediction.shape), 'finite_gradient_tensors': len(grads), 'nonzero_gradient_tensors': nonzero, 'image_ids': list(ids)})
                print('PREFLIGHT STEP', index + 1, json.dumps(steps[-1]), flush=True)
                if index == 2:
                    break
            path = args.run_dir / 'preflight_checkpoint.pth'
            torch.save(model.state_dict(), path)
            model.load_state_dict(torch.load(path, map_location=device), strict=True)
            report = {**state, 'status': 'passed', 'steps': steps, 'seconds': time.monotonic() - started,
                      'peak_allocated_mib': torch.cuda.max_memory_allocated(device) / 2**20,
                      'peak_reserved_mib': torch.cuda.max_memory_reserved(device) / 2**20,
                      'checkpoint_roundtrip': True}
            write_json(state_file, report)
            print('GPU PREFLIGHT PASSED', json.dumps(report), flush=True)
            return

        checkpoint_dir = args.run_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        latest = checkpoint_dir / 'latest.pth'
        start_epoch, best_loss, best_epoch = 1, float('inf'), 0
        for epoch in range(start_epoch, args.epochs + 1):
            state.update(status='training', epoch=epoch, updated_at=datetime.now().isoformat())
            write_json(state_file, state)
            model.train()
            losses = []
            started = time.monotonic()
            lr_used = optimizer.param_groups[0]['lr']
            for step, (x, y, _, _) in enumerate(data['train'], start=1):
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x.to(device))
                loss = criterion(prediction, y.to(device))
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Non-finite loss at epoch {epoch}, step {step}')
                loss.backward()
                optimizer.step()
                losses.append(float(loss))
                if step == 1 or step % 50 == 0:
                    state.update(step=step, steps_per_epoch=len(data['train']), train_loss=float(np.mean(losses)))
                    write_json(state_file, state)
                    print(f'TRAIN epoch={epoch}/{args.epochs} step={step}/{len(data["train"])} loss={np.mean(losses):.6f} lr={lr_used:.8f}', flush=True)
            scheduler.step()
            validation = evaluate(model, data['val'], criterion, device)
            if validation['loss'] < best_loss:
                best_loss, best_epoch = validation['loss'], epoch
                torch.save(model.state_dict(), checkpoint_dir / 'best.pth')
            payload = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                       'scheduler': scheduler.state_dict(), 'best_loss': best_loss, 'best_epoch': best_epoch,
                       'split_sha256': info['split_sha256'], 'model_sha256': info['model_sha256'],
                       'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                       'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(device)}
            payload['loader_rng'] = generator.get_state()
            temp = checkpoint_dir / 'latest.tmp.pth'
            torch.save(payload, temp)
            temp.replace(latest)
            row = {'epoch': epoch, 'train_loss': float(np.mean(losses)), 'val_loss': validation['loss'],
                   'val_global_dice': validation['global']['dice'], 'val_global_iou': validation['global']['iou'],
                   'val_mean_dice': validation['per_image_mean']['dice'], 'lr': lr_used,
                   'seconds': time.monotonic() - started, 'best_epoch': best_epoch}
            history = args.run_dir / 'history.csv'
            exists = history.exists()
            with history.open('a', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
            state.update(epoch_completed=epoch, best_epoch=best_epoch, validation=validation,
                         checkpoint=str(latest), epoch_seconds=row['seconds'])
            write_json(state_file, state)
            print('EPOCH COMPLETE', json.dumps(row), flush=True)
        torch.save(model.state_dict(), checkpoint_dir / 'final.pth')
        model.load_state_dict(torch.load(checkpoint_dir / 'best.pth', map_location=device), strict=True)
        state.update(status='evaluating_test', selected_epoch=best_epoch)
        write_json(state_file, state)
        result = evaluate(model, data['test'], criterion, device, args.run_dir / 'test')
        result.update(selected_epoch=best_epoch, checkpoint=str(checkpoint_dir / 'best.pth'),
                      checkpoint_sha256=hashlib.sha256((checkpoint_dir / 'best.pth').read_bytes()).hexdigest(),
                      seed=args.seed, provenance=info,
                      paper_reference={'iou': 0.9077, 'dice': 0.9516, 'directly_comparable': False})
        write_json(args.run_dir / 'test' / 'summary.json', result)
        global_metrics = result['global']
        text = (f'HAM10000 {args.model} baseline，单次运行 seed={args.seed}\n'
                f'自建病灶分组划分，非作者原始划分；模型参数 {info["parameters"]}\n'
                f'训练 {args.epochs} epochs；按验证损失选择 epoch {best_epoch}\n'
                f'测试图像 {result["images"]} 张\n'
                f'全局 Dice {global_metrics["dice"]*100:.4f}%；IoU {global_metrics["iou"]*100:.4f}%\n'
                f'逐图平均 Dice {result["per_image_mean"]["dice"]*100:.4f}%；IoU {result["per_image_mean"]["iou"]*100:.4f}%\n'
                '论文 MambaLiteUNet 参考：Dice 95.16%，IoU 90.77%；由于划分、代码参数量、运行次数等差异，不能直接视为同协议比较。\n')
        (args.run_dir / 'test' / 'report.txt').write_text(text)
        predicted = len(list((args.run_dir / 'test' / 'predictions').glob('*.png')))
        assert predicted == result['images'] == len(data['test'].dataset)
        write_json(args.run_dir / 'artifact_audit.json', {'complete': True, 'predictions': predicted,
                   'summary': True, 'report': True, 'visualizations': 12, 'selected_epoch': best_epoch})
        state.update(status='complete', test_global=global_metrics, predictions=predicted)
        write_json(state_file, state)
        print(text, flush=True)
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        write_json(state_file, state)
        raise


if __name__ == '__main__':
    main()

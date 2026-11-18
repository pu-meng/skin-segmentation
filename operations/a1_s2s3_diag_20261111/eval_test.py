"""Formal test evaluation for a train_diag run (A1-S2S3 family, e.g. swift-A1-s2s3).

Why this file: train_diag.py only trains and validates; it never touches the test split.
This entry loads the validation-selected best.pth from the run directory and evaluates the
test split once, in the project's standard convention.

Output (one subdir per inference mode):
  <out>/final_test/none/{summary.json,per_image_metrics.csv,predictions/,visualizations/,artifact_audit.json}
  <out>/final_test/mirror4/...   (four views: identity, flip H, flip W, flip H+W)

Metric functions are imported from source/comparison/baseline_runner_seed42.py, the same
ones the A-series uses (raw 0.5 threshold, pooled confusion -> Global Dice / IoU / Acc / Se / Sp,
per-image Dice / IoU / HD95 in pixels).

Usage:
  python eval_test.py --out <run dir> --gpu 0 [--which best] [--limit 3]
"""
import argparse
import csv
import importlib
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

ROOT = '/home/PuMengYu/experiments/skin_segmentation'
sys.path[:0] = [ROOT, ROOT + '/source/comparison', ROOT + '/source/bottleneck_mamba',
                ROOT + '/source/three_module_impl_20261104',
                ROOT + '/source/lightweight_repro_20261111']

from pretrained_unet2d import prepare_imports  # noqa: E402

prepare_imports('mambavision_t')
prepare_imports('tinyvim_s')

from dataset import build_loaders  # noqa: E402

# Copied verbatim from source/comparison/baseline_runner_seed42.py (importing that module
# fails here because source/lightweight_repro_20261111/models.py shadows the `models` package).
from scipy import ndimage  # noqa: E402


def confusion(prediction, target):
    p_, g_ = prediction.astype(bool), target.astype(bool)
    return np.array([(p_ & g_).sum(), (p_ & ~g_).sum(), (~p_ & g_).sum(), (~p_ & ~g_).sum()], dtype=np.int64)


def metrics(counts):
    tp, fp, fn, tn = map(int, counts)

    def ratio(a, b):
        return a / b if b else 0.0

    return {'dice': ratio(2 * tp, 2 * tp + fp + fn), 'iou': ratio(tp, tp + fp + fn),
            'accuracy': ratio(tp + tn, tp + fp + fn + tn), 'sensitivity': ratio(tp, tp + fn),
            'specificity': ratio(tn, tn + fp), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def hd95(prediction, target):
    p_, g_ = prediction.astype(bool), target.astype(bool)
    if not p_.any() or not g_.any():
        return 0.0 if not p_.any() and not g_.any() else float(np.hypot(*p_.shape))
    surface_p = p_ ^ ndimage.binary_erosion(p_)
    surface_g = g_ ^ ndimage.binary_erosion(g_)
    distances = np.concatenate((ndimage.distance_transform_edt(~surface_g)[surface_p],
                                ndimage.distance_transform_edt(~surface_p)[surface_g]))
    return float(np.percentile(distances, 95, method='linear'))

PROCESSED = '/home/PuMengYu/experiments/skin_segmentation/data/processed'
DATASETS = {
    'ISIC2018': dict(
        images_npy=f'{PROCESSED}/isic2018_256_dedup_seed42_20261031/images.npy',
        masks_npy=f'{PROCESSED}/isic2018_256_dedup_seed42_20261031/masks.npy',
        split_csv=f'{PROCESSED}/isic2018_256_dedup_seed42_20261031/split.csv',
        counts={'train': 1780, 'val': 258, 'test': 518}, size=256),
    'HAM10000': dict(
        images_npy=f'{PROCESSED}/ham10000_group42/images.npy',
        masks_npy=f'{PROCESSED}/ham10000_group42/masks.npy',
        split_csv=f'{PROCESSED}/ham10000_group42_dedup_20261023/split.csv',
        counts={'train': 6983, 'val': 1010, 'test': 2020}, size=256),
}
VIEWS = [[], [2], [3], [2, 3]]


def sha256(path):
    import hashlib
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='train_diag run dir (has best.pth / latest.pth)')
    ap.add_argument('--dataset', default='ISIC2018', choices=list(DATASETS))
    ap.add_argument('--candidate', default='source.swift_a1_s2s3_20261111.candidate')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--which', default='best', choices=['best', 'latest'])
    ap.add_argument('--limit', type=int, default=0, help='first N test images only (dry-run)')
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)

    ckpt_path = os.path.join(args.out, f'{args.which}.pth')
    payload = torch.load(ckpt_path, map_location='cpu')
    state = payload['model'] if isinstance(payload, dict) and 'model' in payload else payload
    import os.path as _osp
    epoch = payload.get('epoch') if isinstance(payload, dict) else None
    val_dice = payload.get('val_dice') if isinstance(payload, dict) else None

    build_candidate = importlib.import_module(args.candidate).build_candidate
    model = build_candidate(pretrained=True)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    params = sum(p.numel() for p in model.parameters())

    loader = build_loaders(cfg, 'test', 1, args.workers, False)
    assert len(loader.dataset) == cfg['counts']['test'], (len(loader.dataset), cfg['counts']['test'])
    ids = [str(x) for x in getattr(loader.dataset, 'ids', [])]
    if not ids:
        raise RuntimeError('test loader exposes no ids; refusing to write unattributable products')

    out_root = os.path.join(args.out, 'final_test')
    results = {}
    for mode, name in ((False, 'none'), (True, 'mirror4')):
        target = os.path.join(out_root, name + ('_dryrun' if args.limit else ''))
        os.makedirs(os.path.join(target, 'predictions'), exist_ok=True)
        os.makedirs(os.path.join(target, 'visualizations'), exist_ok=True)
        total = np.zeros(4, dtype=np.int64)
        records, started = [], time.time()
        with torch.no_grad():
            for index, (images, gts, _ids) in enumerate(loader):
                if args.limit and index >= args.limit:
                    break
                images = images.to(device, non_blocking=True)
                prob = model(images)
                if mode:
                    for axes in ([2], [3], [2, 3]):
                        prob = prob + torch.flip(model(torch.flip(images, dims=axes)), dims=axes)
                    prob = prob / len(VIEWS)
                prediction = prob[0, 0].cpu().numpy() >= 0.5
                target_mask = gts[0, 0].numpy() >= 0.5
                count = confusion(prediction, target_mask)
                total += count
                record = {'image_id': ids[index], 'prediction_empty': bool(not prediction.any()),
                          'target_empty': bool(not target_mask.any()), **metrics(count)}
                record['hd95_pixels'] = hd95(prediction, target_mask)
                records.append(record)
                Image.fromarray((prediction.astype(np.uint8) * 255)).save(
                    os.path.join(target, 'predictions', f'{ids[index]}_segmentation.png'))
                if index < 12:
                    rgb = np.clip(images[0].cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
                    gt_rgb = np.repeat(target_mask[..., None].astype(np.uint8) * 255, 3, axis=2)
                    pr_rgb = np.repeat(prediction[..., None].astype(np.uint8) * 255, 3, axis=2)
                    Image.fromarray(np.concatenate((rgb, gt_rgb, pr_rgb), axis=1)).save(
                        os.path.join(target, 'visualizations', f'{ids[index]}_image_gt_prediction.png'))

        summary = {
            'global': metrics(total),
            'images': len(records),
            'per_image_mean': {k: float(np.mean([r[k] for r in records])) for k in ('dice', 'iou', 'hd95_pixels')},
            'hd95_convention': 'pooled bidirectional surface distances; percentile95 linear; 4-neighbor boundary; '
                               'spacing1 pixel; image mean; one-empty penalty sqrt(H^2+W^2)',
            'empty_mask_counts': {
                'prediction_empty': int(sum(r['prediction_empty'] for r in records)),
                'target_empty': int(sum(r['target_empty'] for r in records)),
                'both_empty': int(sum(r['prediction_empty'] and r['target_empty'] for r in records)),
                'prediction_empty_fraction': sum(r['prediction_empty'] for r in records) / max(1, len(records)),
            },
            'dataset': args.dataset,
            'model': 'A1-S2S3 (' + args.candidate + ')',
            'parameters': params,
            'split': 'test',
            'seed': 42,
            'threshold': 0.5,
            'normalisation': 'raw model probability, no per-image min-max',
            'tta': mode,
            'views': VIEWS if mode else [],
            'fusion': 'arithmetic mean of the four view probabilities' if mode else 'single view',
            'checkpoint': ckpt_path,
            'checkpoint_sha256': sha256(ckpt_path),
            'checkpoint_epoch': epoch,
            'checkpoint_val_dice': val_dice,
            'selection': 'max validation Dice (train_diag best.pth)',
            'split_sha256': sha256(cfg['split_csv']),
            'test_used_for_checkpoint_selection': False,
            'evaluated': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'wall_seconds': round(time.time() - started, 1),
        }
        with open(os.path.join(target, 'summary.json'), 'w') as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        with open(os.path.join(target, 'per_image_metrics.csv'), 'w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

        if not args.limit:
            n_pred = len([f for f in os.listdir(os.path.join(target, 'predictions')) if f.endswith('.png')])
            n_vis = len([f for f in os.listdir(os.path.join(target, 'visualizations')) if f.endswith('.png')])
            audit = {
                'complete': bool(n_pred == len(records) == cfg['counts']['test'] and n_vis == 12),
                'split': 'test',
                'prediction_count': n_pred,
                'metric_count': len(records),
                'visualizations': n_vis,
                'confusion_reconciled': None,
                'checkpoint_sha256': summary['checkpoint_sha256'],
                'note': 'counts-only audit by the evaluation entry; pixel-level recomputation not run',
            }
            with open(os.path.join(target, 'artifact_audit.json'), 'w') as handle:
                json.dump(audit, handle, ensure_ascii=False, indent=2)

        results[name] = summary
        print('TEST_DONE', args.dataset, name,
              json.dumps({k: summary[k] for k in ('global', 'per_image_mean', 'images')}, ensure_ascii=False),
              flush=True)

    if not args.limit:
        print('ALL_DONE', args.dataset, 'params', params, 'epoch', epoch, 'val_dice', val_dice, flush=True)


if __name__ == '__main__':
    main()

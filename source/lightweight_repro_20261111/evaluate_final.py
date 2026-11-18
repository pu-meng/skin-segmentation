"""Final test evaluation in OUR metric convention (AD-GBC reproduction).

Vendor convention: iou_score pooled per batch, averaged by batch size, model chosen on val IoU.
Ours: raw sigmoid thresholded at 0.5 (no min-max), pooled confusion -> Global Dice/IoU/Accuracy/
      Sensitivity/Specificity, plus per-image Dice/IoU/HD95. Identical to the A-series convention.

Usage:
  python evaluate_final.py --dataset ISIC2018 --out /media/.../ADGBC_ISIC2018_20261110 --gpu 0
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = '/home/PuMengYu/experiments/skin_segmentation/references/code/AD-GBC'
sys.path.insert(0, VENDOR)
sys.path.insert(0, HERE)

from models import build_model  # noqa: E402

from dataset import build_loaders  # noqa: E402

PROCESSED = '/home/PuMengYu/experiments/skin_segmentation/data/processed'
DATASETS = {
    'HAM10000': dict(
        images_npy=f'{PROCESSED}/ham10000_group42/images.npy',
        masks_npy=f'{PROCESSED}/ham10000_group42/masks.npy',
        split_csv=f'{PROCESSED}/ham10000_group42_dedup_20261023/split.csv',
        counts={'train': 6983, 'val': 1010, 'test': 2020}, size=256),
    'ISIC2018': dict(
        images_npy=f'{PROCESSED}/isic2018_256_dedup_seed42_20261031/images.npy',
        masks_npy=f'{PROCESSED}/isic2018_256_dedup_seed42_20261031/masks.npy',
        split_csv=f'{PROCESSED}/isic2018_256_dedup_seed42_20261031/split.csv',
        counts={'train': 1780, 'val': 258, 'test': 518}, size=256),
}


# Copied verbatim from source/comparison/baseline_runner_seed42.py.
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
        return 0.0 if not p.any() and not g.any() else float(np.hypot(*p.shape))
    surface_p = p ^ ndimage.binary_erosion(p)
    surface_g = g ^ ndimage.binary_erosion(g)
    distances = np.concatenate((ndimage.distance_transform_edt(~surface_g)[surface_p],
                                ndimage.distance_transform_edt(~surface_p)[surface_g]))
    return float(np.percentile(distances, 95, method='linear'))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(DATASETS))
    ap.add_argument('--out', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--limit', type=int, default=0, help='first N test images only (dry-run)')
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu')
    if args.gpu >= 0:
        torch.cuda.set_device(device)

    best_path = os.path.join(args.out, 'best.pth')
    if not os.path.isfile(best_path):
        raise FileNotFoundError(best_path)
    with open(os.path.join(args.out, 'protocol.json')) as handle:
        protocol = json.load(handle)
    arch = protocol['model']

    target_dir = os.path.join(args.out, 'final_test', 'ours' + ('_dryrun' if args.limit else ''))
    os.makedirs(os.path.join(target_dir, 'predictions'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'visualizations'), exist_ok=True)

    model = build_model(arch).to(device)
    state = torch.load(best_path, map_location='cpu')
    model.load_state_dict(state, strict=True)
    model.eval()

    loader = build_loaders(cfg, 'test', 1, args.workers, False)
    assert len(loader.dataset) == cfg['counts']['test'], len(loader.dataset)
    test_ids = loader.dataset.ids

    total = np.zeros(4, dtype=np.int64)
    records = []
    with torch.no_grad():
        for index, (images, gts, _) in enumerate(loader):
            if args.limit and index >= args.limit:
                break
            images = images.to(device, non_blocking=True)
            probability = model(images).sigmoid()[0, 0].cpu().numpy()   # raw sigmoid, no min-max
            prediction = probability >= 0.5
            target = gts[0, 0].numpy() >= 0.5
            count = confusion(prediction, target)
            total += count
            record = {'image_id': test_ids[index], 'prediction_empty': bool(not prediction.any()),
                      'target_empty': bool(not target.any()), **metrics(count)}
            record['hd95_pixels'] = hd95(prediction, target)
            records.append(record)
            Image.fromarray((prediction.astype(np.uint8) * 255)).save(
                os.path.join(target_dir, 'predictions', f'{test_ids[index]}_segmentation.png'))
            if index < 12:
                rgb = np.clip(images[0].cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
                gt_rgb = np.repeat(target[..., None].astype(np.uint8) * 255, 3, axis=2)
                pr_rgb = np.repeat(prediction[..., None].astype(np.uint8) * 255, 3, axis=2)
                Image.fromarray(np.concatenate((rgb, gt_rgb, pr_rgb), axis=1)).save(
                    os.path.join(target_dir, 'visualizations', f'{test_ids[index]}_image_gt_prediction.png'))

    result = {
        'global': metrics(total),
        'images': len(records),
        'per_image_mean': {k: float(np.mean([r[k] for r in records])) for k in ('dice', 'iou', 'hd95_pixels')},
        'hd95_convention': 'pooled bidirectional surface distances; percentile95 linear; 4-neighbor boundary; '
                           'spacing1 pixel; image mean; one-empty penalty sqrt(H^2+W^2)',
        'empty_mask_counts': {
            'prediction_empty': int(sum(r['prediction_empty'] for r in records)),
            'target_empty': int(sum(r['target_empty'] for r in records)),
            'both_empty': int(sum(r['prediction_empty'] and r['target_empty'] for r in records)),
            'prediction_empty_fraction': sum(r['prediction_empty'] for r in records) / len(records),
        },
        'dataset': args.dataset,
        'model': f'{arch} official code (reproduction)',
        'split': 'test',
        'seed': protocol.get('seed', 1029),
        'threshold': 0.5,
        'normalisation': 'raw sigmoid, no per-image min-max (our convention)',
        'tta': False,
        'training_epochs': protocol.get('epochs'),
        'selection': protocol.get('selection'),
        'test_used_for_checkpoint_selection': False,
        'checkpoint': best_path,
        'checkpoint_sha256': sha256(best_path),
        'split_sha256': sha256(cfg['split_csv']),
        'evaluated': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    }
    with open(os.path.join(target_dir, 'summary.json'), 'w') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    with open(os.path.join(target_dir, 'per_image_metrics.csv'), 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    print('OUR_EVAL_DONE', json.dumps({k: result[k] for k in ('dataset', 'images', 'global', 'per_image_mean')},
                                      ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

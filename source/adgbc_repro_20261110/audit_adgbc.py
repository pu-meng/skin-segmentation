"""Recompute the AD-GBC test products from the saved PNGs (audit convention).

Adapted from research-experiment-operations/scripts/audit_final_test_artifacts.py because
this repro's dataset.build_loaders has a different signature (cfg, split, batch, workers,
train_aug) and yields (images, gts, ids) triples, and the model is built from archs_GBC.

Checks: PNG count/names vs split, rebuilt pooled confusion vs summary.global, per-image dice
vs CSV, visualization count, checkpoint sha, and a CPU re-inference of the first N images
(four views when summary.tta is true).

Usage:
  python audit_adgbc.py --product-dir <run>/final_test/ours      --ckpt <run>/best.pth
  python audit_adgbc.py --product-dir <run>/final_test/ours_tta  --ckpt <run>/best.pth
"""
import argparse
import csv
import hashlib
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = '/home/PuMengYu/experiments/skin_segmentation/references/code/AD-GBC'
sys.path.insert(0, VENDOR)
sys.path.insert(0, HERE)

import archs_GBC  # noqa: E402
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


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='ISIC2018', choices=list(DATASETS))
    ap.add_argument('--product-dir', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--reinfer', type=int, default=3)
    ap.add_argument('--tol-frac', type=float, default=1e-4)
    ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()

    import pandas as pd

    cfg = DATASETS[args.dataset]
    target = args.product_dir
    summary = json.load(open(os.path.join(target, 'summary.json')))
    protocol = json.load(open(os.path.join(os.path.dirname(os.path.dirname(target)), 'protocol.json')))
    arch = protocol['arch']

    frame = pd.read_csv(cfg['split_csv'])
    test_ids = frame.loc[frame['split'] == 'test', 'image_id'].astype(str).tolist()

    pred_dir = os.path.join(target, 'predictions')
    png_files = sorted(f for f in os.listdir(pred_dir) if f.endswith('.png'))
    expected = {f'{i}_segmentation.png' for i in test_ids}
    prediction_count = len(png_files)
    names_ok = set(png_files) == expected

    rows = list(csv.DictReader(open(os.path.join(target, 'per_image_metrics.csv'))))
    metric_count = len(rows)
    csv_dice = {r['image_id']: float(r['dice']) for r in rows}

    loader = build_loaders(cfg, 'test', 1, args.workers, False)
    loader_ids = [str(x) for x in loader.dataset.ids]
    ids_order_ok = loader_ids == test_ids

    total = np.zeros(4, dtype=np.int64)
    dice_diffs, shape_ok = [], True
    for index, (images, gts, _ids) in enumerate(loader):
        image_id = test_ids[index]
        saved = np.array(Image.open(os.path.join(pred_dir, f'{image_id}_segmentation.png')).convert('L'))
        prediction, target_mask = saved > 127, gts[0, 0].numpy() >= 0.5
        if prediction.shape != target_mask.shape:
            shape_ok = False
            continue
        p, g = prediction, target_mask
        total += np.array([(p & g).sum(), (p & ~g).sum(), (~p & g).sum(), (~p & ~g).sum()], dtype=np.int64)
        denom = 2 * (p & g).sum() + (p & ~g).sum() + (~p & g).sum()
        if denom and image_id in csv_dice:
            dice_diffs.append(abs(float(2 * (p & g).sum() / denom) - csv_dice[image_id]))

    sm = summary['global']
    reported = np.array([sm['tp'], sm['fp'], sm['fn'], sm['tn']], dtype=np.int64)
    confusion_reconciled = bool(np.array_equal(total, reported))

    vis_dir = os.path.join(target, 'visualizations')
    visualizations = len([f for f in os.listdir(vis_dir) if f.endswith('.png')]) if os.path.isdir(vis_dir) else 0
    ckpt_sha = sha256(args.ckpt)
    checkpoint_match = ckpt_sha == summary.get('checkpoint_sha256')

    recheck, tolerance = [], max(1, int(cfg['size'] * cfg['size'] * args.tol_frac))
    if args.reinfer:
        device = torch.device('cpu')
        model = archs_GBC.__dict__[arch](
            num_classes=1, input_channels=3, deep_supervision=False,
            gbc_num_balls=protocol.get('gbc_num_balls', 32), gbc_proj_dim=None,
            use_diag_cov=True, tau=protocol.get('tau', 1.0))
        model.load_state_dict(torch.load(args.ckpt, map_location='cpu'), strict=True)
        model = model.to(device).eval()
        with torch.no_grad():
            for index, (images, gts, _ids) in enumerate(build_loaders(cfg, 'test', 1, args.workers, False)):
                if index >= args.reinfer:
                    break
                logits = model(images)
                if summary.get('tta'):
                    for axes in ([2], [3], [2, 3]):
                        logits = logits + torch.flip(model(torch.flip(images, dims=axes)), dims=axes)
                    logits = logits / 4
                pred = logits.sigmoid()[0, 0].numpy() >= 0.5
                saved = np.array(Image.open(os.path.join(
                    pred_dir, f'{test_ids[index]}_segmentation.png')).convert('L')) > 127
                recheck.append({'image_id': test_ids[index], 'differing_pixels': int((pred ^ saved).sum())})

    complete = bool(
        prediction_count == metric_count == len(test_ids)
        and names_ok and ids_order_ok and shape_ok and confusion_reconciled
        and max(dice_diffs, default=1.0) < 1e-6
        and visualizations == 12
        and checkpoint_match
        and all(item['differing_pixels'] <= tolerance for item in recheck)
    )

    audit = {
        'complete': complete,
        'split': 'test',
        'prediction_count': prediction_count,
        'metric_count': metric_count,
        'visualizations': visualizations,
        'confusion_reconciled': confusion_reconciled,
        'checkpoint_sha256': ckpt_sha,
        'max_abs_dice_diff_vs_csv': max(dice_diffs) if dice_diffs else None,
        'reported_confusion': reported.tolist(),
        'rebuilt_confusion': total.tolist(),
        'names_match_split': names_ok,
        'loader_ids_match_split': ids_order_ok,
        'shape_match_gt': shape_ok,
        'checkpoint_sha256_matches_summary': checkpoint_match,
        'cpu_reinference': recheck,
        'cpu_reinference_tolerance_pixels': tolerance,
        'cpu_reinference_note': 'GPU/CPU fp32 accumulation order differs; a few pixels at the 0.5 threshold may flip. Every diff is recorded above.',
    }
    out = os.path.join(target, 'artifact_audit.json')
    json.dump(audit, open(out, 'w'), ensure_ascii=False, indent=2)
    print('AUDIT_WRITTEN', out, 'complete=', complete,
          'confusion_reconciled=', confusion_reconciled,
          'max_dice_diff=', audit['max_abs_dice_diff_vs_csv'], flush=True)


if __name__ == '__main__':
    main()

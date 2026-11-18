"""Ablation evaluation for the A1-S2S3 V1 diagnostic checkpoints.

Six runtime modes, no retraining, no weight changes, no touching the original residual path:
  all           every correction active
  no_skip2      Skip2 returns (S, D) unchanged
  no_skip3      Skip3 returns (S, D) unchanged
  no_both       both skips off
  no_cross      QKV cross-gating term zeroed (local conv still active)
  no_local      local convolution term zeroed (cross-gating still active)

Reports validation Global Dice, validation loss, per-image Dice, and the delta against `all`.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

ROOT = '/home/PuMengYu/experiments/skin_segmentation'
sys.path[:0] = [ROOT, ROOT + '/source/comparison', ROOT + '/source/bottleneck_mamba',
                ROOT + '/source/three_module_impl_20261104',
                ROOT + '/source/lightweight_repro_20261111']

from pretrained_unet2d import prepare_imports  # noqa: E402

prepare_imports('mambavision_t')
prepare_imports('tinyvim_s')

from dataset import build_loaders  # noqa: E402

PROCESSED = '/home/PuMengYu/experiments/skin_segmentation/data/processed'
DATASETS = {
    'ISIC2018': dict(
        images_npy=f'{PROCESSED}/isic2018_256_dedup_seed42_20261031/images.npy',
        masks_npy=f'{PROCESSED}/isic2018_256_dedup_seed42_20261031/masks.npy',
        split_csv=f'{PROCESSED}/isic2018_256_dedup_seed42_20261031/split.csv',
        counts={'train': 1780, 'val': 258, 'test': 518}, size=256),
}
from source.a1_s2s3_20261111.candidate import build_candidate  # noqa: E402

MODES = ('all', 'no_skip2', 'no_skip3', 'no_both', 'no_cross', 'no_local')


def set_mode(model, mode):
    sk2, sk3 = model.decoder.skip2, model.decoder.skip3
    for blk in (sk2, sk3):
        blk.runtime = {'enabled': True, 'use_gate': True, 'use_local': True}
    if mode == 'no_skip2':
        sk2.runtime['enabled'] = False
    elif mode == 'no_skip3':
        sk3.runtime['enabled'] = False
    elif mode == 'no_both':
        sk2.runtime['enabled'] = False
        sk3.runtime['enabled'] = False
    elif mode == 'no_cross':
        for blk in (sk2, sk3):
            blk.runtime['use_gate'] = False
    elif mode == 'no_local':
        for blk in (sk2, sk3):
            blk.runtime['use_local'] = False


@torch.no_grad()
def evaluate(model, loader, device):
    inter = psum = gsum = 0.0
    bce_total, seen = 0.0, 0
    per_image = []
    bce = torch.nn.BCEWithLogitsLoss(reduction='none')
    for images, gts, _ in loader:
        images = images.to(device, non_blocking=True)
        gts = gts.to(device, non_blocking=True)
        logits = model(images)
        bce_total += float(bce(logits, gts).mean().item()) * images.size(0)
        seen += images.size(0)
        pred = (logits.sigmoid() >= 0.5)
        gt = (gts >= 0.5)
        inter += float((pred & gt).sum())
        psum += float(pred.sum())
        gsum += float(gt.sum())
        for b in range(pred.shape[0]):
            p = pred[b].sum().item()
            g = gt[b].sum().item()
            i = (pred[b] & gt[b]).sum().item()
            per_image.append((2 * i + 1e-6) / (p + g + 1e-6))
    return {
        'global_dice': (2 * inter + 1e-6) / (psum + gsum + 1e-6),
        'per_image_dice': float(np.mean(per_image)),
        'bce': bce_total / seen,
        'per_image_values': per_image,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='diagnostic run dir')
    ap.add_argument('--gpu', type=int, default=1)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    cfg = DATASETS['ISIC2018']
    loader = build_loaders(cfg, 'val', args.batch_size, args.workers, False)
    print(f'validation images {len(loader.dataset)}', flush=True)

    model = build_candidate(pretrained=True).to(device).eval()

    weights = sorted(f for f in os.listdir(args.out) if f.startswith('weights_epoch') and f.endswith('.pth'))
    results = []
    for weight_file in weights:
        tag = weight_file.replace('weights_epoch', '').replace('.pth', '')
        payload = torch.load(os.path.join(args.out, weight_file), map_location='cpu')
        model.load_state_dict(payload['model'], strict=True)
        model.eval()
        base = None
        for mode in MODES:
            set_mode(model, mode)
            metrics = evaluate(model, loader, device)
            if mode == 'all':
                base = metrics
            row = {'weights': tag, 'mode': mode,
                   'global_dice': metrics['global_dice'],
                   'per_image_dice': metrics['per_image_dice'],
                   'bce': metrics['bce'],
                   'd_global_dice': metrics['global_dice'] - base['global_dice'],
                   'd_per_image_dice': metrics['per_image_dice'] - base['per_image_dice']}
            results.append(row)
            print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()},
                             ensure_ascii=False), flush=True)

        # per-case delta between `all` and each closed mode, for the last weight only
        if tag == weights[-1].replace('weights_epoch', '').replace('.pth', ''):
            set_mode(model, 'all')
            all_vals = evaluate(model, loader, device)['per_image_values']
            for mode in ('no_skip2', 'no_skip3', 'no_cross', 'no_local'):
                set_mode(model, mode)
                vals = evaluate(model, loader, device)['per_image_values']
                deltas = [v - a for v, a in zip(vals, all_vals)]
                with open(os.path.join(args.out, f'per_case_delta_{mode}.csv'), 'w', newline='') as handle:
                    writer = csv.writer(handle)
                    writer.writerow(['index', 'dice_all', f'dice_{mode}', 'delta'])
                    for i, (a, v, d) in enumerate(zip(all_vals, vals, deltas)):
                        writer.writerow([i, f'{a:.6f}', f'{v:.6f}', f'{d:.6f}'])

    keys = list(results[0].keys())
    with open(os.path.join(args.out, 'ablation_results.csv'), 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print('ABLATION_DONE', flush=True)


if __name__ == '__main__':
    main()

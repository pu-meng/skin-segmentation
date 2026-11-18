"""300-epoch training for the Skip2/Skip3 ablation arms on one dataset, no per-branch diagnostics.

Why a separate entry: train_diag.py spends time every epoch snapshotting skip-branch gradients
and updates, which only makes sense while debugging the S2S3 block itself. The ablation arms
just need the same optimisation/schedule/loss as the existing A1-S2S3_ISIC300 run, so this file
copies that protocol and drops the diagnostics.

Protocol (identical to A1-S2S3_ISIC300_20261111 / train_diag.py defaults):
  AdamW(lr=1e-3, wd=0.01, betas=(0.9,0.999), eps=1e-8), CosineAnnealingLR(T_max=300, eta_min=1e-5),
  BceDiceLoss(wb=1, wd=1, smooth=1) on probabilities, batch 8, 256x256, seed 42.
Per epoch it writes diag_log.csv; every 10 epochs weights_epochN.pth + latest.pth (resume-ready),
and best.pth on validation Global Dice.

Usage:
  python train_abl.py --out <dir> --dataset ISIC2018 --gpu 0 --epochs 300 \
      --candidate source.a1_s2s3_20261111.candidate [--no-skip2|--no-skip3|--no-skips]
"""
import argparse
import csv
import importlib
import json
import os
import sys
import time

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
    'HAM10000': dict(
        images_npy=f'{PROCESSED}/ham10000_group42/images.npy',
        masks_npy=f'{PROCESSED}/ham10000_group42/masks.npy',
        split_csv=f'{PROCESSED}/ham10000_group42_dedup_20261023/split.csv',
        counts={'train': 6983, 'val': 1010, 'test': 2020}, size=256),
}


class OfficialBceDiceLoss(torch.nn.Module):
    """Mirror of references/code/MambaLiteUNet/utils.py BceDiceLoss (same as train_diag.py).

    BCELoss expects probabilities, which is what these models return (forward applies sigmoid).
    """

    def __init__(self, wb=1.0, wd=1.0):
        super().__init__()
        self.bce = torch.nn.BCELoss()
        self.wb, self.wd = wb, wd

    def forward(self, pred, target):
        bce = self.bce(pred, target)
        size = pred.size(0)
        flat_p = pred.view(size, -1)
        flat_t = target.view(size, -1)
        inter = (flat_p * flat_t).sum(1)
        dice = 1 - ((2 * inter + 1) / (flat_p.sum(1) + flat_t.sum(1) + 1)).sum() / size
        return self.wd * dice + self.wb * bce


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--dataset', default='ISIC2018', choices=list(DATASETS))
    ap.add_argument('--candidate', default='source.a1_s2s3_20261111.candidate')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=0.01)
    ap.add_argument('--t-max', type=int, default=300)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--no-skips', action='store_true', help='both Skip2 and Skip3 restored to A1')
    ap.add_argument('--no-skip3', action='store_true', help='only Skip3 restored to A1')
    ap.add_argument('--no-skip2', action='store_true', help='only Skip2 restored to A1')
    ap.add_argument('--variant', default='none',
                    help='candidate-specific switch, e.g. no_skip1 / no_e2 / no_d2d3')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    cfg = DATASETS[args.dataset]
    import inspect

    build_candidate = importlib.import_module(args.candidate).build_candidate
    sig = inspect.signature(build_candidate)
    kwargs = {'pretrained': True}
    if 'use_skip2' in sig.parameters:
        kwargs['use_skip2'] = not (args.no_skips or args.no_skip2)
        kwargs['use_skip3'] = not (args.no_skips or args.no_skip3)
    if 'variant' in sig.parameters:
        kwargs['variant'] = args.variant
    model = build_candidate(**kwargs).to(device)
    total_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                  betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max, eta_min=1e-5)
    criterion = OfficialBceDiceLoss().to(device)

    train_loader = build_loaders(cfg, 'train', args.batch_size, args.workers, True)
    val_loader = build_loaders(cfg, 'val', args.batch_size, args.workers, False)
    print(f'model params {total_params:,} candidate {args.candidate} variant {args.variant} '
          f'skip2={not (args.no_skips or args.no_skip2)} skip3={not (args.no_skips or args.no_skip3)} '
          f'train {len(train_loader.dataset)} val {len(val_loader.dataset)} steps/epoch {len(train_loader)}',
          flush=True)

    def save_state(tag, epoch, val_dice):
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'epoch': epoch, 'val_dice': val_dice},
                   os.path.join(args.out, f'{tag}.pth'))

    def dump_csv(rows):
        if not rows:
            return
        with open(os.path.join(args.out, 'diag_log.csv'), 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    rows, start_epoch, best_dice = [], 1, -1.0
    resume_path = os.path.join(args.out, 'latest.pth')
    if args.resume and os.path.exists(resume_path):
        ck = torch.load(resume_path, map_location=device)
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        start_epoch = ck['epoch'] + 1
        best_dice = ck.get('val_dice', -1.0)
        print(f'resumed from epoch {ck["epoch"]} (val_dice {best_dice:.6f})', flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        started = time.time()
        model.train()
        running, seen = 0.0, 0
        for images, gts, _ in train_loader:
            images = images.to(device, non_blocking=True)
            gts = gts.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, gts)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * images.size(0)
            seen += images.size(0)
        scheduler.step()
        train_loss = running / max(1, seen)

        model.eval()
        v_loss, inter, psum, gsum = 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for images, gts, _ in val_loader:
                images = images.to(device, non_blocking=True)
                gts = gts.to(device, non_blocking=True)
                logits = model(images)
                v_loss += float(criterion(logits, gts).item()) * images.size(0)
                pred = (logits >= 0.5).float()
                inter += float((pred * gts).sum())
                psum += float(pred.sum())
                gsum += float(gts.sum())
        val_loss = v_loss / len(val_loader.dataset)
        val_dice = (2 * inter + 1e-6) / (psum + gsum + 1e-6)

        peak_mb = (torch.cuda.max_memory_allocated(device) / 1024 ** 2) if device.type == 'cuda' else 0.0
        row = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
               'peak_gpu_mb': round(peak_mb, 1), 'val_global_dice': val_dice,
               'lr': optimizer.param_groups[0]['lr'], 'seconds': round(time.time() - started, 1)}
        rows.append(row)
        print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()},
                         ensure_ascii=False), flush=True)
        dump_csv(rows)
        if epoch % 10 == 0 or epoch == args.epochs:
            torch.save({'model': model.state_dict()},
                       os.path.join(args.out, f'weights_epoch{epoch}.pth'))
            save_state('latest', epoch, val_dice)
        if val_dice > best_dice:
            best_dice = val_dice
            save_state('best', epoch, val_dice)

    dump_csv(rows)
    print('ABL_DONE best_val_dice=%.6f params=%d' % (best_dice, total_params), flush=True)


if __name__ == '__main__':
    main()

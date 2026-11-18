"""AD-GBC (CVPR 2026) reproduction on our data.

Vendor settings are copied verbatim from `references/code/AD-GBC/train_GBC.py`:
  arch GBC_Rolling_Unet_{S,M,L} | epochs 400 | batch 8 | Adam (others lr 1e-4, gbc lr 1e-2,
  weight_decay 1e-4 both) | CosineAnnealingLR(T_max=epochs, eta_min=1e-5) | no grad clip |
  loss BCEDiceWithGeometryLoss(div_weight=0.01, scale_weight=0.01) | K=32 | tau=1.0 |
  use_diag_cov True | input 256 | model selection = best validation IoU.

Only two things change: the data split is ours, and test is never touched during training.
Final metrics come from evaluate_final.py in our own convention.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim import lr_scheduler

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = '/home/PuMengYu/experiments/skin_segmentation/references/code/AD-GBC'
sys.path.insert(0, VENDOR)   # lower priority
sys.path.insert(0, HERE)     # our dataset.py shadows the vendor one (vendor is unused here)

import archs_GBC  # noqa: E402
import losses  # noqa: E402
from metrics import iou_score  # noqa: E402

from dataset import build_loaders  # noqa: E402

PROCESSED = '/home/PuMengYu/experiments/skin_segmentation/data/processed'
DATASETS = {
    'HAM10000': dict(
        images_npy=f'{PROCESSED}/ham10000_group42/images.npy',
        masks_npy=f'{PROCESSED}/ham10000_group42/masks.npy',
        split_csv=f'{PROCESSED}/ham10000_group42_dedup_20261023/split.csv',
        counts={'train': 6983, 'val': 1010, 'test': 2020}),
    'ISIC2018': dict(
        images_npy=f'{PROCESSED}/isic2018_256_dedup_seed42_20261031/images.npy',
        masks_npy=f'{PROCESSED}/isic2018_256_dedup_seed42_20261031/masks.npy',
        split_csv=f'{PROCESSED}/isic2018_256_dedup_seed42_20261031/split.csv',
        counts={'train': 1780, 'val': 258, 'test': 518}),
}
DATASETS['HAM10000']['size'] = 256
DATASETS['ISIC2018']['size'] = 256


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(DATASETS))
    ap.add_argument('--arch', default='GBC_Rolling_Unet_L', choices=['GBC_Rolling_Unet_S',
                                                                   'GBC_Rolling_Unet_M',
                                                                   'GBC_Rolling_Unet_L'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=400)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--gbc_lr', type=float, default=1e-2)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--div_weight', type=float, default=0.01)
    ap.add_argument('--scale_weight', type=float, default=0.01)
    ap.add_argument('--gbc_num_balls', type=int, default=32)
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--min_lr', type=float, default=1e-5)
    ap.add_argument('--seed', type=int, default=1029)
    ap.add_argument('--smoke', type=int, default=0, help='stop after N epochs (pipeline test)')
    return ap.parse_args()


def seed_torch(seed):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def write_status(out, **payload):
    payload['updated'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    with open(os.path.join(out, 'status.json'), 'w') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    cfg = DATASETS[args.dataset]
    seed_torch(args.seed)
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    os.makedirs(args.out, exist_ok=True)

    protocol = dict(
        paper='AD-GBC: Anisotropic Granular-Ball Skip-Connection Refiner for UNet-Based '
              'Medical Image Segmentation (CVPR 2026)',
        vendor_repo='https://github.com/SiaShen-dot/AD-GBC', vendor_commit='25abdca',
        arch=args.arch, dataset=args.dataset, input_size=cfg['size'],
        epochs=args.epochs, batch_size=args.batch_size, optimizer='Adam',
        lr=args.lr, gbc_lr=args.gbc_lr, weight_decay=args.weight_decay,
        scheduler=f'CosineAnnealingLR(T_max={args.epochs}, eta_min={args.min_lr})',
        loss='BCEDiceWithGeometryLoss(div_weight=%.3f, scale_weight=%.3f)'
             % (args.div_weight, args.scale_weight),
        gbc_num_balls=args.gbc_num_balls, tau=args.tau, use_diag_cov=True,
        grad_clip=None, seed=args.seed,
        augmentation='RandomRotate90 + HorizontalFlip',
        selection='best validation IoU (vendor convention)',
        test_policy='never touched during training; evaluated once by evaluate_final.py',
        deviations=[
            'Data split is ours (HAM 6983/1010/2020, ISIC 1780/258/518) instead of 80/20 random.',
            'Vendor dataset.py divides by 255 and albumentations Normalize divides by 255 again, '
            'collapsing input to about -2.11+/-0.003; we normalise once with ImageNet mean/std.',
            'Vendor trains on 2000-image ISIC17; we run on our HAM10000 and ISIC2018 caches.',
        ],
        counts=cfg['counts'], split_csv=cfg['split_csv'],
        paper_report='ISIC17 IoU 84.95 / F1 91.72 (Rolling-UNet-L), 80/20 split, validation metrics, 3 seeds',
    )
    with open(os.path.join(args.out, 'protocol.json'), 'w') as handle:
        json.dump(protocol, handle, ensure_ascii=False, indent=2)

    criterion = losses.BCEDiceWithGeometryLoss(div_weight=args.div_weight,
                                               scale_weight=args.scale_weight).to(device)
    model = archs_GBC.__dict__[args.arch](
        num_classes=1, input_channels=3, deep_supervision=False,
        gbc_num_balls=args.gbc_num_balls, gbc_proj_dim=None,
        use_diag_cov=True, tau=args.tau).to(device)

    gbc_params, other_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lowered = name.lower()
        (gbc_params if ('centers' in lowered or 'gbc' in lowered) else other_params).append(param)
    groups = [{'params': other_params, 'lr': args.lr, 'weight_decay': args.weight_decay}]
    if gbc_params:
        groups.append({'params': gbc_params, 'lr': args.gbc_lr, 'weight_decay': args.weight_decay})
    optimizer = optim.Adam(groups, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    train_loader = build_loaders(cfg, 'train', args.batch_size, args.workers, True)
    val_loader = build_loaders(cfg, 'val', args.batch_size, args.workers, False)
    assert len(train_loader.dataset) == cfg['counts']['train'], len(train_loader.dataset)
    assert len(val_loader.dataset) == cfg['counts']['val'], len(val_loader.dataset)

    parameters = sum(p.numel() for p in model.parameters())
    print(f'params={parameters:,} train={len(train_loader.dataset)} '
          f'val={len(val_loader.dataset)} gbc_params={sum(p.numel() for p in gbc_params):,}',
          flush=True)

    history_path = os.path.join(args.out, 'history.csv')
    with open(history_path, 'w', newline='') as handle:
        handle.write('epoch,train_loss,train_iou,train_dice,val_loss,val_iou,val_dice,'
                     'lr,seconds,best_epoch\n')

    best_iou, best_epoch = 0.0, -1
    write_status(args.out, status='running', pid=os.getpid(), dataset=args.dataset,
                 arch=args.arch, epoch=0, epochs=args.epochs, best_iou=None, best_epoch=None)

    for epoch in range(args.epochs):
        started = time.time()

        model.train()
        losses_sum, ious_sum, dices_sum, seen = 0.0, 0.0, 0.0, 0
        for images, gts, _ in train_loader:
            images = images.to(device, non_blocking=True)
            gts = gts.to(device, non_blocking=True)
            output = model(images)
            loss = criterion(output, gts, model)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            iou, dice = iou_score(output, gts)
            losses_sum += float(loss.item()) * images.size(0)
            ious_sum += float(iou) * images.size(0)
            dices_sum += float(dice) * images.size(0)
            seen += images.size(0)
        train_loss, train_iou, train_dice = losses_sum / seen, ious_sum / seen, dices_sum / seen
        lr_now = optimizer.param_groups[0]['lr']
        scheduler.step()

        model.eval()
        v_loss, v_iou, v_dice, v_seen = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for images, gts, _ in val_loader:
                images = images.to(device, non_blocking=True)
                gts = gts.to(device, non_blocking=True)
                output = model(images)
                loss = criterion(output, gts, model)
                iou, dice = iou_score(output, gts)
                v_loss += float(loss.item()) * images.size(0)
                v_iou += float(iou) * images.size(0)
                v_dice += float(dice) * images.size(0)
                v_seen += images.size(0)
        val_loss, val_iou, val_dice = v_loss / v_seen, v_iou / v_seen, v_dice / v_seen

        if val_iou > best_iou:
            best_iou, best_epoch = val_iou, epoch
            torch.save(model.state_dict(), os.path.join(args.out, 'best.pth'))
        torch.save(model.state_dict(), os.path.join(args.out, 'last.pth'))

        seconds = time.time() - started
        with open(history_path, 'a', newline='') as handle:
            handle.write(f'{epoch},{train_loss:.6f},{train_iou:.6f},{train_dice:.6f},'
                         f'{val_loss:.6f},{val_iou:.6f},{val_dice:.6f},{lr_now:.8f},'
                         f'{seconds:.2f},{best_epoch}\n')
        print(f'epoch {epoch + 1}/{args.epochs} train_loss={train_loss:.4f} '
              f'val_iou={val_iou:.4f} val_dice={val_dice:.4f} best={best_iou:.4f}@{best_epoch} '
              f'{seconds:.1f}s', flush=True)
        write_status(args.out, status='running', pid=os.getpid(), dataset=args.dataset,
                     arch=args.arch, epoch=epoch + 1, epochs=args.epochs,
                     best_iou=best_iou, best_epoch=best_epoch, seconds=round(seconds, 1))

        if args.smoke and epoch + 1 >= args.smoke:
            print('SMOKE_DONE', flush=True)
            break

    with open(os.path.join(args.out, 'completion.json'), 'w') as handle:
        json.dump({'best_iou': best_iou, 'best_epoch': best_epoch, 'parameters': parameters,
                   'epochs_run': epoch + 1, 'smoke': bool(args.smoke),
                   'finished': time.strftime('%Y-%m-%dT%H:%M:%S%z')},
                  handle, ensure_ascii=False, indent=2)
    write_status(args.out, status='complete', pid=os.getpid(), dataset=args.dataset,
                 arch=args.arch, epoch=epoch + 1, epochs=args.epochs,
                 best_iou=best_iou, best_epoch=best_epoch)
    print(f'BEST val_iou={best_iou:.4f} @epoch {best_epoch}', flush=True)


if __name__ == '__main__':
    main()

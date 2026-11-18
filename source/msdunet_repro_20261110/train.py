"""MSDUNet reproduction on our own splits.

Protocol decided with the user:
  - data split: our split.csv (identical to the A-series experiments)
  - selection: validation Dice (author's metric), test never touched during training
  - input size, loss, optimizer, lr schedule, epochs, batch, clipping: official code defaults
  - augmentation: follows --augmentation default (off)

Usage:
  python train.py --dataset HAM10000 --gpu 0 --out /media/.../MSDUNET_HAM_20261110
  python train.py --dataset ISIC2018 --gpu 1 --out ... --smoke   # 2 epochs pipeline check
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from compat import build_msdunet
from dataset import build_loaders

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


def structure_loss(pred, mask):
    """Official MSDUNet loss: boundary-weighted BCE + boundary-weighted IoU."""
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, decay_rate=0.1, decay_epoch=200, epoch=0):
    """Official semantics: multiply the current lr by decay_rate**(epoch//decay_epoch)."""
    decay = decay_rate ** (epoch // decay_epoch)
    for group in optimizer.param_groups:
        group['lr'] *= decay


@torch.no_grad()
def evaluate(model, loader, device):
    """Official metric: per-image Dice with smooth=1, after per-image min-max normalisation."""
    model.eval()
    dices = []
    for images, gts in loader:
        images = images.to(device, non_blocking=True)
        out = model(images)
        out = F.interpolate(out, size=gts.shape[-2:], mode='bilinear', align_corners=False)
        prob = out.sigmoid().float().cpu().numpy()[:, 0]
        target = gts.numpy()[:, 0]
        for b in range(prob.shape[0]):
            p = prob[b]
            p = (p - p.min()) / (p.max() - p.min() + 1e-8)
            pred = p >= 0.5
            gt = target[b] >= 0.5
            inter = np.logical_and(pred, gt).sum()
            dices.append(float((2 * inter + 1) / (pred.sum() + gt.sum() + 1)))
    return float(np.mean(dices)), dices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(DATASETS))
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--size', type=int, default=352)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--clip', type=float, default=0.5)
    ap.add_argument('--decay_rate', type=float, default=0.1)
    ap.add_argument('--decay_epoch', type=int, default=200)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--smoke', action='store_true', help='2 epochs, no full run')
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)

    train_loader = build_loaders(cfg, 'train', args.size, args.batch, args.workers, augment=False)
    val_loader = build_loaders(cfg, 'val', args.size, 1, args.workers)
    for split, loader in (('train', train_loader), ('val', val_loader)):
        got, want = len(loader.dataset), cfg['counts'][split]
        assert got == want, f'{split} count {got} != expected {want}'

    model = build_msdunet(num_classes=1).to(device)
    params = sum(p.numel() for p in model.parameters())
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total_updates = 0
    epochs = 2 if args.smoke else args.epochs

    status_path = os.path.join(args.out, 'status.json')

    def write_status(state, extra=None):
        payload = dict(status=state, pid=os.getpid(), dataset=args.dataset,
                       out=args.out, updated=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
        if extra:
            payload.update(extra)
        with open(status_path, 'w') as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    write_status('train')

    meta = dict(dataset=args.dataset, params=params, input_size=args.size, batch=args.batch,
                epochs=epochs, lr=args.lr, wd=args.wd, clip=args.clip,
                selection='max validation Dice', loss='official structure_loss',
                split_csv=cfg['split_csv'], counts=cfg['counts'], smoke=args.smoke,
                started=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    with open(os.path.join(args.out, 'run_config.json'), 'w') as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    print('CONFIG', json.dumps(meta, ensure_ascii=False), flush=True)

    history_path = os.path.join(args.out, 'history.csv')
    if not os.path.exists(history_path):
        with open(history_path, 'w', newline='') as fh:
            csv.writer(fh).writerow(['epoch', 'train_loss', 'val_dice', 'val_loss', 'lr', 'seconds', 'best_epoch'])

    best_dice, best_epoch = -1.0, 0
    best_path = os.path.join(args.out, 'best.pth')
    for epoch in range(1, epochs + 1):
        if epoch > 1:
            adjust_lr(optimizer, args.decay_rate, args.decay_epoch, epoch=epoch - 1)
        model.train()
        started = time.time()
        running, steps = 0.0, 0
        for images, gts in train_loader:
            images = images.to(device, non_blocking=True)
            gts = gts.to(device, non_blocking=True)
            out = model(images)
            loss = structure_loss(out, gts)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_gradient(optimizer, args.clip)
            optimizer.step()
            running += float(loss.detach())
            steps += 1
            total_updates += 1
            if steps % 50 == 0:
                print(f'TRAIN epoch={epoch}/{epochs} step={steps}/{len(train_loader)} loss={running/steps:.4f}', flush=True)
        train_loss = running / max(steps, 1)

        val_dice, _ = evaluate(model, val_loader, device)
        with torch.no_grad():
            model.eval()
            val_losses = []
            for images, gts in val_loader:
                images = images.to(device)
                gts = gts.to(device)
                val_losses.append(float(structure_loss(model(images), gts)))
            val_loss = float(np.mean(val_losses))

        improved = val_dice > best_dice
        if improved:
            best_dice, best_epoch = val_dice, epoch
            torch.save(model.state_dict(), best_path)
        torch.save(model.state_dict(), os.path.join(args.out, 'last.pth'))

        seconds = time.time() - started
        lr_now = optimizer.param_groups[0]['lr']
        with open(history_path, 'a', newline='') as fh:
            csv.writer(fh).writerow([epoch, f'{train_loss:.6f}', f'{val_dice:.6f}', f'{val_loss:.6f}',
                                     f'{lr_now:.8f}', f'{seconds:.2f}', best_epoch])
        print(f'EPOCH {epoch} train_loss={train_loss:.4f} val_dice={val_dice:.6f} '
              f'val_loss={val_loss:.4f} best={best_dice:.6f}@{best_epoch} {seconds:.1f}s '
              f'lr={lr_now:.2e} {"*" if improved else ""}', flush=True)

    result = dict(dataset=args.dataset, epochs=epochs, best_epoch=best_epoch,
                  best_val_dice=best_dice, params=params, smoke=args.smoke,
                  finished=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    with open(os.path.join(args.out, 'train_summary.json'), 'w') as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print('TRAIN_DONE', json.dumps(result, ensure_ascii=False), flush=True)

    if args.smoke:
        write_status('complete', dict(best_epoch=best_epoch, best_val_dice=best_dice, smoke=True))
        print('smoke mode: skipping test evaluation', flush=True)
        return

    # test is evaluated exactly once, after training finished, using the validation-selected best
    test_loader = build_loaders(cfg, 'test', args.size, 1, args.workers)
    assert len(test_loader.dataset) == cfg['counts']['test']
    model.load_state_dict(torch.load(best_path, map_location='cpu'), strict=True)
    model.to(device)
    test_dice, per_image = evaluate(model, test_loader, device)
    summary = dict(result, test_dice=test_dice, test_images=len(per_image),
                   selection='max validation Dice', test_used_for_selection=False,
                   threshold=0.5, metric='per-image Dice mean, smooth=1, min-max normalised')
    with open(os.path.join(args.out, 'final_test.json'), 'w') as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, 'per_image_dice.csv'), 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['index', 'dice'])
        for i, value in enumerate(per_image):
            writer.writerow([i, f'{value:.6f}'])
    write_status('complete', dict(best_epoch=best_epoch, best_val_dice=best_dice,
                                  test_dice=test_dice, test_images=len(per_image)))
    print('TEST_DONE', json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

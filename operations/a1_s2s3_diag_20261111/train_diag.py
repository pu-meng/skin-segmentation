"""10-epoch short diagnostic for A1-S2S3 V1.

Not a real training run: 10 epochs on ISIC2018 to see whether Skip2/Skip3 actually learn
and whether closing a branch changes validation predictions.

Protocol copied from the A1 ISIC2018 protocol (ISIC2018_A0_A1_FRESH300_20261108):
  AdamW(lr=1e-3, wd=0.01, betas=(0.9,0.999), eps=1e-8), CosineAnnealingLR(T_max=300),
  BceDiceLoss(wb=1,wd=1,smooth=1), batch 8, 256x256, seed 42, val batch 8.
Only the epoch count changes (10 instead of 300), so these 10 epochs are the first 10 of a
would-be full run; the LR schedule is deliberately left at T_max=300.

Per epoch it records: alpha/beta/rho means, the norm ratio of the applied correction to the
input feature, gradient norms per branch, and actual parameter update norms per branch.
Checkpoints at epoch 0 (initial weights), 5 and 10 feed the ablation study.
"""
import argparse
import csv
import json
import math
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
from models import BceDiceLoss  # noqa: E402
import importlib  # noqa: E402

PROCESSED = '/home/PuMengYu/experiments/skin_segmentation/data/processed'
DATASETS = {
    'ISIC2018': dict(
        images_npy=f'{PROCESSED}/isic2018_256_dedup_seed42_20261031/images.npy',
        masks_npy=f'{PROCESSED}/isic2018_256_dedup_seed42_20261031/masks.npy',
        split_csv=f'{PROCESSED}/isic2018_256_dedup_seed42_20261031/split.csv',
        counts={'train': 1780, 'val': 258, 'test': 518}, size=256),
}

def branches_of(blk):
    """V2 content block: 7x7 depthwise + 4C bottleneck; local is a single depthwise."""
    return ('content_s.dw', 'content_s.pw_in', 'content_s.pw_out',
            'gate_s', 'local_s')


def norm(t):
    return float(t.detach().float().norm())


def module_of(root, path):
    node = root
    for part in path.split('.'):
        node = getattr(node, part)
    return node


def grad_norm(module):
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().float().norm()) ** 2
    return math.sqrt(total)


def snapshot(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def update_norm(module, before):
    total = 0.0
    after = module.state_dict()
    for k, v in after.items():
        if k in before and v.dtype.is_floating_point:
            total += float((v.detach() - before[k]).float().norm()) ** 2
    return math.sqrt(total)


def dice_from_logits(logits, target):
    # The model's forward already applies sigmoid; threshold the output directly.
    pred = (logits >= 0.5)
    gt = (target >= 0.5)
    inter = (pred & gt).sum().item()
    denom = pred.sum().item() + gt.sum().item()
    return (2 * inter + 1e-6) / (denom + 1e-6), inter, pred.sum().item(), gt.sum().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--gpu', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=0.01)
    ap.add_argument('--t-max', type=int, default=300)
    ap.add_argument('--no-skips', action='store_true',
                    help='A1 control: neither Skip2 nor Skip3 uses the new module')
    ap.add_argument('--candidate', default='source.a1_s2s3_20261111.candidate',
                    help='module exposing build_candidate(...)')
    ap.add_argument('--resume', action='store_true',
                    help='continue from latest.pth in --out')
    ap.add_argument('--no-skip2', action='store_true',
                    help='V1-S2: Skip2 restored to the original A1 concat')
    ap.add_argument('--no-skip3', action='store_true',
                    help='V1-S3: Skip3 restored to the original A1 concat')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    torch.manual_seed(42)

    cfg = DATASETS['ISIC2018']
    build_candidate = importlib.import_module(args.candidate).build_candidate
    model = build_candidate(
        pretrained=True,
        use_skip2=not (args.no_skips or args.no_skip2),
        use_skip3=not (args.no_skips or args.no_skip3),
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max, eta_min=1e-5)
    class OfficialBceDiceLoss(torch.nn.Module):
        """Mirror of references/code/MambaLiteUNet/utils.py BceDiceLoss.

        BCELoss expects probabilities, which is exactly what the A1 model's forward returns
        (it applies sigmoid internally). Using a logits loss here would sigmoid twice.
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

    criterion = OfficialBceDiceLoss().to(device)

    def bce_dice(logits, target):
        # Same wb=1, wd=1, smooth=1 loss the A1 runs use; keeps gradients intact.
        return criterion(logits, target)

    train_loader = build_loaders(cfg, 'train', args.batch_size, args.workers, True)
    val_loader = build_loaders(cfg, 'val', args.batch_size, args.workers, False)
    print(f'model params {total_params:,} train {len(train_loader.dataset)} '
          f'val {len(val_loader.dataset)} steps/epoch {len(train_loader)}', flush=True)

    skip2 = model.decoder.skip2
    skip3 = model.decoder.skip3
    for blk in (skip2, skip3):
        if blk is not None:
            blk.collect_stats = True

    def save(tag):
        path = os.path.join(args.out, f'weights_epoch{tag}.pth')
        torch.save({'model': model.state_dict()}, path)
        return path

    def save_state(tag, epoch, val_dice):
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'val_dice': val_dice},
                   os.path.join(args.out, f'{tag}.pth'))

    def dump_csv(rows):
        if not rows:
            return
        with open(os.path.join(args.out, 'diag_log.csv'), 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    rows = []
    start_epoch = 1
    best_dice = -1.0
    resume_path = os.path.join(args.out, 'latest.pth')
    if args.resume and os.path.exists(resume_path):
        ck = torch.load(resume_path, map_location=device)
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        start_epoch = ck['epoch'] + 1
        best_dice = ck.get('val_dice', -1.0)
        print(f'resumed from epoch {ck["epoch"]} (val_dice {best_dice:.6f})', flush=True)
    if start_epoch == 1:
        save(0)

    for epoch in range(start_epoch, args.epochs + 1):
        started = time.time()
        model.train()
        # snapshot branch parameters to measure real updates
        before = {}
        for level, blk in (('skip2', skip2), ('skip3', skip3)):
            if blk is None:
                continue
            for path in branches_of(blk):
                before[f'{level}.{path}'] = snapshot(module_of(blk, path))
            before[f'{level}.alpha_s'] = skip_scalar_snapshot(blk.alpha_s)
            before[f'{level}.alpha_d'] = skip_scalar_snapshot(blk.alpha_d)

        running, seen = 0.0, 0
        grad_accum = {}
        for images, gts, _ in train_loader:
            images = images.to(device, non_blocking=True)
            gts = gts.to(device, non_blocking=True)
            logits = model(images)
            loss = bce_dice(logits, gts)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for level, blk in (('skip2', skip2), ('skip3', skip3)):
                if blk is None:
                    continue
                for path in branches_of(blk):
                    key = f'{level}.{path}'
                    grad_accum[key] = grad_accum.get(key, 0.0) + grad_norm(module_of(blk, path))
            running += float(loss.item()) * images.size(0)
            seen += images.size(0)
            optimizer.step()
        scheduler.step()
        train_loss = running / seen

        # branch gradients averaged over the epoch's steps
        steps = len(train_loader)
        grads = {k: v / steps for k, v in grad_accum.items()}
        updates = {}
        for level, blk in (('skip2', skip2), ('skip3', skip3)):
            if blk is None:
                continue
            for path in branches_of(blk):
                key = f'{level}.{path}'
                updates[key] = update_norm(module_of(blk, path), before[key])
            updates[f'{level}.alpha_s'] = float((blk.alpha_s.detach() - before[f'{level}.alpha_s']).abs().max())
            updates[f'{level}.alpha_d'] = float((blk.alpha_d.detach() - before[f'{level}.alpha_d']).abs().max())

        # validation with a fixed batch to capture stats + global dice
        model.eval()
        v_loss, inter, psum, gsum = 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for images, gts, _ in val_loader:
                images = images.to(device, non_blocking=True)
                gts = gts.to(device, non_blocking=True)
                logits = model(images)
                v_loss += float(bce_dice(logits, gts).item()) * images.size(0)
                _, i_, p_, g_ = dice_from_logits(logits, gts)
                inter += i_
                psum += p_
                gsum += g_
        val_loss = v_loss / len(val_loader.dataset)
        val_dice = (2 * inter + 1e-6) / (psum + gsum + 1e-6)

        stats = {}
        for level, blk in (('skip2', skip2), ('skip3', skip3)):
            if blk is None:
                continue
            st = blk.stats or {}
            for k, v in st.items():
                stats[f'{level}.{k}'] = v

        peak_mb = (torch.cuda.max_memory_allocated(device) / 1024 ** 2) if device.type == 'cuda' else 0.0
        row = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'peak_gpu_mb': round(peak_mb, 1),
               'val_global_dice': val_dice, 'lr': optimizer.param_groups[0]['lr'],
               'seconds': round(time.time() - started, 1), **stats,
               **{f'grad.{k}': v for k, v in grads.items()},
               **{f'upd.{k}': v for k, v in updates.items()}}
        rows.append(row)
        print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()},
                         ensure_ascii=False), flush=True)

        dump_csv(rows)
        if epoch % 10 == 0 or epoch == args.epochs:
            save(epoch)
            save_state('latest', epoch, val_dice)
        if val_dice > best_dice:
            best_dice = val_dice
            save_state('best', epoch, val_dice)

    dump_csv(rows)
    print('DIAG_DONE best_val_dice=%.6f' % best_dice, flush=True)


def skip_scalar_snapshot(t):
    return t.detach().clone()


if __name__ == '__main__':
    main()

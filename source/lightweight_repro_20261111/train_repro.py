"""Unified reproduction trainer for the two lightweight baselines (CMUNeXt / FuseUNet).

Protocol (identical to the A-series so the numbers are comparable):
  input 256x256, batch 8, 300 epochs, RandomRotate90 + HorizontalFlip, BceDiceLoss,
  best checkpoint = minimum validation loss, test set never touched.

Optimiser follows each paper:
  cmunext : SGD(lr 0.01, momentum 0.9, wd 1e-4), poly decay (1 - it/max_it)^0.9
  fuseunet: SGD(lr 0.01, momentum 0.99, wd 3e-5), poly decay (nnUNet-2D defaults, 300 epochs)
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dataset import build_loaders  # noqa: E402
from models import BceDiceLoss, build_model, dice_score  # noqa: E402

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

OPT = {
    'cmunext': dict(lr=0.01, momentum=0.9, wd=1e-4, power=0.9,
                    paper='CMUNeXt, ISBI 2024 oral (official main.py: SGD 0.01, momentum 0.9, wd 1e-4, poly 0.9)'),
    'fuseunet': dict(lr=0.01, momentum=0.99, wd=3e-5, power=0.9,
                     paper='FuseUNet, nnUNet 2-D defaults (SGD 0.01, momentum 0.99, wd 3e-5, poly 0.9)'),
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['cmunext', 'fuseunet'])
    ap.add_argument('--dataset', required=True, choices=list(DATASETS))
    ap.add_argument('--out', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--smoke', type=int, default=0, help='stop after N epochs (pipeline test)')
    return ap.parse_args()


def write_status(out, **payload):
    path = os.path.join(out, 'status.json')
    tmp = path + '.tmp'
    with open(tmp, 'w') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def main():
    args = parse_args()
    cfg = DATASETS[args.dataset]
    opt_cfg = OPT[args.model]
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    os.makedirs(args.out, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    model = build_model(args.model).to(device)
    parameters = sum(p.numel() for p in model.parameters())

    protocol = dict(
        model=args.model,
        paper=opt_cfg['paper'],
        dataset=args.dataset, input_size=cfg['size'], epochs=args.epochs,
        batch_size=args.batch_size,
        optimizer=f"SGD(lr={opt_cfg['lr']}, momentum={opt_cfg['momentum']}, wd={opt_cfg['wd']})",
        scheduler=f"poly (1 - iter/max_iter)^{opt_cfg['power']}",
        loss='BceDiceLoss(wb=1, wd=1, smooth=1)  # same as the A-series',
        augmentation='RandomRotate90 + HorizontalFlip (same as the A-series)',
        selection='minimum validation loss',
        test_policy='never touched during training; evaluated once by evaluate_final.py',
        counts=cfg['counts'], split_csv=cfg['split_csv'], parameters=parameters,
        seed=args.seed,
        deviations=[
            'Data split is ours (HAM 6983/1010/2020, ISIC 1780/258/518), not the paper split.',
            '300 epochs for both models so the numbers are comparable with the A-series; '
            'the papers use their own schedules (CMUNeXt 300, nnUNet 1000).',
            'Loss/augmentation/split are unified with the A-series to isolate the architecture.',
        ],
    )
    with open(os.path.join(args.out, 'protocol.json'), 'w') as handle:
        json.dump(protocol, handle, ensure_ascii=False, indent=2)

    criterion = BceDiceLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=opt_cfg['lr'],
                                momentum=opt_cfg['momentum'], weight_decay=opt_cfg['wd'])

    train_loader = build_loaders(cfg, 'train', args.batch_size, args.workers, True)
    val_loader = build_loaders(cfg, 'val', args.batch_size, args.workers, False)
    assert len(train_loader.dataset) == cfg['counts']['train']
    assert len(val_loader.dataset) == cfg['counts']['val']

    steps_per_epoch = len(train_loader)
    total_steps = max(1, steps_per_epoch * args.epochs)
    print(f'model={args.model} params={parameters:,} train={len(train_loader.dataset)} '
          f'val={len(val_loader.dataset)} steps/epoch={steps_per_epoch}', flush=True)

    history_path = os.path.join(args.out, 'history.csv')
    with open(history_path, 'w', newline='') as handle:
        handle.write('epoch,train_loss,train_dice,val_loss,val_dice,lr,seconds,best_epoch\n')

    best_loss, best_epoch = float('inf'), -1
    global_step = 0
    write_status(args.out, status='running', pid=os.getpid(), dataset=args.dataset,
                 model=args.model, epoch=0, epochs=args.epochs, path=args.out)

    for epoch in range(args.epochs):
        started = time.time()
        model.train()
        tr_loss, tr_dice, seen = 0.0, 0.0, 0
        for images, gts, _ in train_loader:
            images = images.to(device, non_blocking=True)
            gts = gts.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, gts)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            lr_now = opt_cfg['lr'] * (1.0 - global_step / total_steps) ** opt_cfg['power']
            for group in optimizer.param_groups:
                group['lr'] = lr_now

            tr_loss += float(loss.item()) * images.size(0)
            tr_dice += dice_score(logits, gts) * images.size(0)
            seen += images.size(0)

        model.eval()
        v_loss, v_dice, v_seen = 0.0, 0.0, 0
        with torch.no_grad():
            for images, gts, _ in val_loader:
                images = images.to(device, non_blocking=True)
                gts = gts.to(device, non_blocking=True)
                logits = model(images)
                v_loss += float(criterion(logits, gts).item()) * images.size(0)
                v_dice += dice_score(logits, gts) * images.size(0)
                v_seen += images.size(0)

        train_loss, train_dice = tr_loss / seen, tr_dice / seen
        val_loss, val_dice = v_loss / v_seen, v_dice / v_seen

        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            torch.save(model.state_dict(), os.path.join(args.out, 'best.pth'))
        torch.save(model.state_dict(), os.path.join(args.out, 'last.pth'))

        seconds = time.time() - started
        with open(history_path, 'a', newline='') as handle:
            handle.write(f'{epoch + 1},{train_loss:.6f},{train_dice:.6f},{val_loss:.6f},'
                         f'{val_dice:.6f},{lr_now:.8f},{seconds:.2f},{best_epoch + 1}\n')
        write_status(args.out, status='running', pid=os.getpid(), dataset=args.dataset,
                     model=args.model, epoch=epoch + 1, epochs=args.epochs,
                     val_loss=val_loss, val_dice=val_dice, best_epoch=best_epoch + 1,
                     seconds=seconds, path=args.out)
        print(f'epoch {epoch + 1}/{args.epochs} loss={train_loss:.4f} val_loss={val_loss:.4f} '
              f'val_dice={val_dice:.4f} best={best_epoch + 1} {seconds:.1f}s', flush=True)

        if args.smoke and (epoch + 1) >= args.smoke:
            print('SMOKE_STOP', flush=True)
            break

    write_status(args.out, status='complete', pid=os.getpid(), dataset=args.dataset,
                 model=args.model, epochs=args.epochs, best_epoch=best_epoch + 1,
                 best_val_loss=best_loss, path=args.out)
    print('TRAIN_COMPLETE', json.dumps(dict(model=args.model, dataset=args.dataset,
                                            best_epoch=best_epoch + 1,
                                            best_val_loss=best_loss), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

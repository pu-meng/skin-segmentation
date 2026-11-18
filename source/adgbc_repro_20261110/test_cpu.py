"""CPU acceptance test for the AD-GBC adapter (no GPU, no training)."""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = '/home/PuMengYu/experiments/skin_segmentation/references/code/AD-GBC'
sys.path.insert(0, VENDOR)
sys.path.insert(0, HERE)

import archs_GBC
import losses
from metrics import iou_score
from dataset import build_loaders

CFG = dict(
    images_npy='/home/PuMengYu/experiments/skin_segmentation/data/processed/isic2018_256_dedup_seed42_20261031/images.npy',
    masks_npy='/home/PuMengYu/experiments/skin_segmentation/data/processed/isic2018_256_dedup_seed42_20261031/masks.npy',
    split_csv='/home/PuMengYu/experiments/skin_segmentation/data/processed/isic2018_256_dedup_seed42_20261031/split.csv',
    size=256,
)

print('=== 1. parameters ===')
for name, cls in [('S', archs_GBC.GBC_Rolling_Unet_S), ('M', archs_GBC.GBC_Rolling_Unet_M),
                  ('L', archs_GBC.GBC_Rolling_Unet_L)]:
    model = cls(num_classes=1, input_channels=3, deep_supervision=False,
                gbc_num_balls=32, gbc_proj_dim=None, use_diag_cov=True, tau=1.0)
    total = sum(p.numel() for p in model.parameters())
    gbc = sum(p.numel() for n, p in model.named_parameters()
              if 'centers' in n.lower() or 'gbc' in n.lower())
    print(f'  {name}: total={total:,} ({total / 1e6:.2f}M)  gbc_group={gbc:,}')

print('=== 2. dataset ===')
loader = build_loaders(CFG, 'train', 2, 0, True)
images, gts, meta = next(iter(loader))
print(f'  image shape={tuple(images.shape)} dtype={images.dtype} '
      f'min={images.min():.3f} max={images.max():.3f} mean={images.mean():.3f} std={images.std():.3f}')
print(f'  gt shape={tuple(gts.shape)} unique={torch.unique(gts).tolist()} ids={meta["img_id"]}')
assert tuple(images.shape) == (2, 3, 256, 256)
assert set(torch.unique(gts).tolist()) <= {0.0, 1.0}

print('=== 3. forward / loss / backward (S, train mode) ===')
model = archs_GBC.GBC_Rolling_Unet_S(num_classes=1, input_channels=3, deep_supervision=False,
                                     gbc_num_balls=32, gbc_proj_dim=None, use_diag_cov=True, tau=1.0)
criterion = losses.BCEDiceWithGeometryLoss(div_weight=0.01, scale_weight=0.01)
model.train()
output = model(images)
assert isinstance(output, tuple) and len(output) == 2, type(output)
seg, inter = output
print(f'  train forward -> seg {tuple(seg.shape)}, intermediates keys={sorted(inter)}')
loss = criterion(output, gts, model)
loss.backward()
grads = [(n, float(p.grad.abs().max())) for n, p in model.named_parameters()
         if p.grad is not None and ('centers' in n.lower() or 'log_sigma' in n.lower())]
print(f'  loss={float(loss):.4f}  total={float(loss):.4f}')
for name, value in grads:
    print(f'    grad {name}: max={value:.3e}')
assert all(value > 0 for _, value in grads), 'GBC parameters received no gradient'
iou, dice = iou_score(seg, gts)
print(f'  train-mode iou_score iou={float(iou):.4f} dice={float(dice):.4f}')

print('=== 4. eval mode returns a tensor ===')
model.eval()
with torch.no_grad():
    out_eval = model(images)
print(f'  eval forward -> {type(out_eval).__name__} {tuple(out_eval.shape)}')
assert torch.is_tensor(out_eval)

print('=== 5. L model forward on CPU (shape only, no backward) ===')
big = archs_GBC.GBC_Rolling_Unet_L(num_classes=1, input_channels=3, deep_supervision=False,
                                   gbc_num_balls=32, gbc_proj_dim=None, use_diag_cov=True, tau=1.0)
big.eval()
with torch.no_grad():
    out_big = big(torch.randn(1, 3, 256, 256))
print(f'  L output {tuple(out_big.shape)}')
assert tuple(out_big.shape) == (1, 1, 256, 256)

print('ALL_CHECKS_PASSED')

from typing import Tuple, List

import torch
import numpy as np
from medpy.metric import dc, jc


def compute_batch_avearge_metrics(
    output_dis: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[float, float, float]:
    """
    Compute three evaluation metrics (dice_soft, dice_hard, iou) for a batch (Sample Index Average).

    Calculation:
    - dice_soft: Mean of background and foreground Dice (using medpy.dc, based on binary masks).
    - dice_hard: Foreground Dice (using medpy.dc).
    - iou: Foreground IoU (using medpy.jc).

    Args:
        output_dis (torch.Tensor): Predicted binary masks (N, 1, H, W)
        targets (torch.Tensor): Ground truth binary masks (N, 1, H, W)

    Returns:
        Tuple[float, float, float]: dice_soft, dice_hard, iou
    """
    # Shape: [N, 1, H, W] -> [N, H, W]
    pred_label = output_dis.squeeze(1)
    gt_label = targets.squeeze(1)

    # Convert to numpy
    pred_np = pred_label.detach().cpu().numpy()
    gt_np = gt_label.detach().cpu().numpy()

    # Binary masks (foreground/background)
    pred_fg = (pred_np == 1)
    pred_bg = (pred_np == 0)
    gt_fg = (gt_np == 1)
    gt_bg = (gt_np == 0)

    # medpy metrics (overall calculation in N,H,W dimension → pixel-level micro-average)
    dice_bg = float(dc(pred_bg, gt_bg))
    dice_fg = float(dc(pred_fg, gt_fg))
    dice_soft = (dice_bg + dice_fg) / 2.0

    iou_fg = float(jc(pred_fg, gt_fg))
    dice_hard = dice_fg

    return dice_soft, dice_hard, iou_fg


def compute_sample_average_metrics(
    output_dis: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[float, float, float]:
    """
    Compute three evaluation metrics (dice_soft, dice_hard, iou) for a batch (Sample Index Average).

    Calculation:
    - dice_soft: Mean of background and foreground Dice (using medpy.dc, based on binary masks).
    - dice_hard: Foreground Dice (using medpy.dc).
    - iou: Foreground IoU (using medpy.jc).

    Args:
        output_dis (torch.Tensor): Predicted binary masks (N, 1, H, W)
        targets (torch.Tensor): Ground truth binary masks (N, 1, H, W)

    Returns:
        Tuple[float, float, float]: (dice_soft_mean, dice_hard_mean, iou_mean)
    """
    # [N, 1, H, W] -> [N, H, W]
    pred_label = output_dis.squeeze(1)
    gt_label = targets.squeeze(1)

    # Convert to numpy
    pred_np = pred_label.detach().cpu().numpy()
    gt_np = gt_label.detach().cpu().numpy()

    # Binary masks (foreground/background)
    pred_fg = (pred_np == 1)
    pred_bg = (pred_np == 0)
    gt_fg = (gt_np == 1)
    gt_bg = (gt_np == 0)

    num_samples = pred_np.shape[0]
    if num_samples == 0:
        return 0.0, 0.0, 0.0

    dice_soft_vals: List[float] = []
    dice_hard_vals: List[float] = []
    iou_vals: List[float] = []

    # Calculate in sample-level and then average
    for i in range(num_samples):
        dice_bg_i = float(dc(pred_bg[i], gt_bg[i]))
        dice_fg_i = float(dc(pred_fg[i], gt_fg[i]))
        dice_soft_i = (dice_bg_i + dice_fg_i) / 2.0
        iou_fg_i = float(jc(pred_fg[i], gt_fg[i]))

        dice_soft_vals.append(dice_soft_i)
        dice_hard_vals.append(dice_fg_i)
        iou_vals.append(iou_fg_i)

    return float(np.mean(dice_soft_vals)), float(np.mean(dice_hard_vals)), float(np.mean(iou_vals))

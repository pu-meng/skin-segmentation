from typing import Tuple, Optional, List
import os
import csv

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from source.tools.isic_tools.metrics import compute_sample_average_metrics
from source.utils.log import Logger
from source.config import TrainConfig


def valid_polyp_epoch(
    now_epoch: int,
    valid_loader: torch.utils.data.DataLoader,
    eval_model: torch.nn.Module,
    validation_results_csv: Optional[str] = None,
    writer: Optional[SummaryWriter] = None,
    logger: Optional[Logger] = None,
    base_model_path: Optional[str] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    topk_checkpoints: Optional[List[Tuple[int, str, float]]] = None,
    orig_model: Optional[torch.nn.Module] = None,
    ema_model: Optional[torch.nn.Module] = None,
    args: Optional[TrainConfig] = None,
) -> Tuple[float, float, float]:
    """
    在验证集上验证模型性能（完全参考 ISIC 的实现风格）。

    指标计算：
    - 使用 compute_sample_average_metrics 计算 batch 内部的样本平均 SoftDice/HardDice/IoU，
      再按样本数量加权累计，得到验证集整体的平均值。
    - 复合分数 composite_score = 0.8 * HardDice + 0.15 * IoU + 0.05 * SoftDice。

    返回 (Avg_IoU, Avg_HardDice, Composite_Score)
    """
    eval_model.eval()

    dice_soft_sum = 0.0
    dice_hard_sum = 0.0
    iou_sum = 0.0
    sample_count = 0

    with torch.no_grad():
        for images, targets in tqdm(
            valid_loader,
            desc=f"Epoch {now_epoch} Validation",
            total=len(valid_loader),
            mininterval=0.001
        ):
            images = images.float().cuda()
            targets = targets.float().cuda()

            logits = eval_model(images)

            if isinstance(logits, (list, tuple)):
                logits = logits[0] if len(logits) == 3 else logits[-1]

            if logits.dim() == 4 and logits.size(1) >= 2:
                logits = torch.softmax(logits, dim=1)
                pred_mask = torch.argmax(logits, dim=1, keepdim=True).float()
            else:
                prob = torch.sigmoid(logits)
                pred_mask = (prob > 0.5).float()

            batch_soft, batch_hard, batch_iou = compute_sample_average_metrics(pred_mask, targets)
            batch_size = int(targets.size(0))
            dice_soft_sum += float(batch_soft) * batch_size
            dice_hard_sum += float(batch_hard) * batch_size
            iou_sum += float(batch_iou) * batch_size
            sample_count += batch_size

    if sample_count > 0:
        dice_soft_mean = dice_soft_sum / sample_count
        dice_hard_mean = dice_hard_sum / sample_count
        iou_mean = iou_sum / sample_count
    else:
        dice_soft_mean = 0.0
        dice_hard_mean = 0.0
        iou_mean = 0.0

    composite_score = 0.8 * dice_hard_mean + 0.15 * iou_mean + 0.05 * dice_soft_mean
    net_score = composite_score

    # 写 CSV（与 ISIC 保持一致的列名与格式）
    if validation_results_csv is None:
        base_dir = base_model_path if (base_model_path is not None and os.path.isdir(base_model_path)) else "."
        validation_results_csv = os.path.join(base_dir, "validation_results.csv")

    file_exists = os.path.exists(validation_results_csv)
    if not file_exists:
        with open(validation_results_csv, 'w', newline='', encoding="utf-8") as csvfile:
            csv_writer = csv.writer(csvfile)
            headers = ['Epoch', 'Avg_SoftDice', 'Avg_HardDice', 'Avg_IoU', 'Composite_Score']
            csv_writer.writerow(headers)

    with open(validation_results_csv, 'a', newline='', encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        row = [
            f"Validation Epoch {now_epoch}",
            f"{dice_soft_mean:.6f}",
            f"{dice_hard_mean:.6f}",
            f"{iou_mean:.6f}",
            f"{float(net_score):.6f}",
        ]
        csv_writer.writerow(row)

    if writer is not None:
        writer.add_scalar("valid/dice_soft", torch.tensor(dice_soft_mean), now_epoch)
        writer.add_scalar("valid/dice_hard", torch.tensor(dice_hard_mean), now_epoch)
        writer.add_scalar("valid/iou", torch.tensor(iou_mean), now_epoch)
        writer.add_scalar("valid/composite_score", torch.tensor(net_score), now_epoch)

    if logger is not None:
        logger.log_metrics({
            'type': 'validation',
            'epoch': now_epoch,
            'dice_soft': float(dice_soft_mean),
            'dice_hard': float(dice_hard_mean),
            'iou': float(iou_mean),
            'composite_score': float(net_score),
        })

    # Top-K 保存策略（与 ISIC 一致，按需启用）
    if (
        base_model_path is not None and
        topk_checkpoints is not None and
        optimizer is not None
    ):
        checkpoint_path = os.path.join(
            base_model_path,
            f"validation_epoch_{now_epoch:03d}_composite_{float(net_score):.6f}.pth.tar",
        )

        should_save = False
        max_k = max(1, int(getattr(args, 'top_k', 3))) if args is not None else 3
        if len(topk_checkpoints) < max_k:
            should_save = True
        elif len(topk_checkpoints) > 0 and float(net_score) > topk_checkpoints[-1][2]:
            worst = topk_checkpoints[-1]
            if os.path.exists(worst[1]):
                os.remove(worst[1])
            topk_checkpoints.pop()
            should_save = True

        if should_save:
            if logger is not None:
                logger.log_metrics({
                    'type': 'checkpoint_saved',
                    'epoch': now_epoch,
                    'checkpoint': os.path.basename(checkpoint_path),
                    'composite_score': float(net_score),
                })

            if getattr(args, 'use_ema', False) and ema_model is not None and hasattr(ema_model, 'module'):
                state_dict = ema_model.module.state_dict()
            elif orig_model is not None:
                state_dict = orig_model.state_dict()
            else:
                state_dict = eval_model.state_dict()

            torch.save(
                {
                    "epoch": now_epoch,
                    "state_dict": state_dict,
                    "opt_dict": optimizer.state_dict(),
                    "valid_score": float(net_score),
                    "composite_score": float(net_score),
                },
                checkpoint_path,
            )

            topk_checkpoints.append((now_epoch, checkpoint_path, float(net_score)))
            topk_checkpoints.sort(key=lambda x: x[2], reverse=True)

    return iou_mean, dice_hard_mean, float(net_score)

from typing import Tuple, List, Optional
import os
import csv

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from source.utils.log import Logger
from source.config import TrainConfig
from .metrics import compute_sample_average_metrics
from torch.utils.data import DataLoader
from source.data.dataset.isic_dataset import ISICDataset


def valid_isic_epoch(
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
    Validate the model on the validation set.

    Metrics calculation:
        - Use compute_sample_average_metrics to calculate the average SoftDice/HardDice/IoU within a batch,
          and then average by the number of samples to get the average validation set.
        - Composite score composite_score = 0.8 * HardDice + 0.15 * IoU + 0.05 * SoftDice.

    Args:
        now_epoch (int): Current epoch
        valid_loader (torch.utils.data.DataLoader): Validation data loader
        eval_model (torch.nn.Module): Evaluation model
        validation_results_csv (Optional[str]): Validation results CSV path
        writer (Optional[SummaryWriter]): TensorBoard writer
        logger (Optional[Logger]): Logger
        base_model_path (Optional[str]): Base model path
        optimizer (Optional[torch.optim.Optimizer]): Optimizer
        topk_checkpoints (Optional[List[Tuple[int, str, float]]]): Top-K checkpoints
        orig_model (Optional[torch.nn.Module]): Original model
        ema_model (Optional[torch.nn.Module]): EMA model
        args (Optional[TrainConfig]): Training arguments

    Returns:
        (float, float, float): (Avg_IoU, Avg_HardDice, Composite_Score)
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

    debug_mode = bool(getattr(args, 'debug_mode', False))
    test_soft_mean = None
    test_hard_mean = None
    test_iou_mean = None
    ph2_soft_mean = None
    ph2_hard_mean = None
    ph2_iou_mean = None

    if debug_mode:
        dataset_name = getattr(args, 'dataset_name', 'ISIC2017')
        if str(dataset_name) == 'ISIC2018':
            isic_debug_test_folder = './data/ISIC2018_test_npy_224_224_debug'
        else:
            isic_debug_test_folder = './data/ISIC2017_test_npy_224_224_debug'
        ph2_debug_folder = './data/PH2_npy_224_224_debug'

        num_workers = int(getattr(args, 'workers', 4))
        eval_batch_size = int(getattr(args, 'batch_size', 1))
        common_kwargs = {'pin_memory': True, 'num_workers': num_workers}

        test_debug_loader = DataLoader(
            dataset=ISICDataset(dataset_folder=isic_debug_test_folder, dataset_type='test', cache_data=False),
            batch_size=eval_batch_size,
            shuffle=False,
            **common_kwargs
        )
        ph2_debug_loader = DataLoader(
            dataset=ISICDataset(dataset_folder=ph2_debug_folder, dataset_type='test', cache_data=False),
            batch_size=eval_batch_size,
            shuffle=False,
            **common_kwargs
        )

        def _evaluate_loader_metrics(loader: DataLoader) -> Tuple[float, float, float]:
            soft_sum = 0.0
            hard_sum = 0.0
            iou_sum_local = 0.0
            count = 0
            with torch.no_grad():
                for img_b, tgt_b in loader:
                    img_b = img_b.float().cuda()
                    tgt_b = tgt_b.float().cuda()
                    out_b = eval_model(img_b)
                    if out_b.dim() == 4 and out_b.size(1) >= 2:
                        out_b = torch.softmax(out_b, dim=1)
                        pred_b = torch.argmax(out_b, dim=1, keepdim=True).float()
                    else:
                        prob_b = torch.sigmoid(out_b)
                        pred_b = (prob_b > 0.5).float()
                    s, h, iou_local = compute_sample_average_metrics(pred_b, tgt_b)
                    bsz = int(tgt_b.size(0))
                    soft_sum += float(s) * bsz
                    hard_sum += float(h) * bsz
                    iou_sum_local += float(iou_local) * bsz
                    count += bsz
            if count > 0:
                return soft_sum / count, hard_sum / count, iou_sum_local / count
            return 0.0, 0.0, 0.0

        test_soft_mean, test_hard_mean, test_iou_mean = _evaluate_loader_metrics(test_debug_loader)
        ph2_soft_mean, ph2_hard_mean, ph2_iou_mean = _evaluate_loader_metrics(ph2_debug_loader)

    if validation_results_csv is None:
        base_dir = base_model_path if (base_model_path is not None and os.path.isdir(base_model_path)) else "."
        validation_results_csv = os.path.join(base_dir, "validation_results.csv")

    reported_score = float(net_score)

    file_exists = os.path.exists(validation_results_csv)
    if not file_exists:
        with open(validation_results_csv, 'w', newline='', encoding="utf-8") as csvfile:
            csv_writer = csv.writer(csvfile)
            headers = ['Epoch', 'Avg_SoftDice', 'Avg_HardDice', 'Avg_IoU', 'Composite_Score']
            if debug_mode:
                headers.extend([
                    'Test_SoftDice', 'Test_HardDice', 'Test_IoU',
                    'PH2_SoftDice', 'PH2_HardDice', 'PH2_IoU'
                ])
            csv_writer.writerow(headers)

    with open(validation_results_csv, 'a', newline='', encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        row = [
            f"Validation Epoch {now_epoch}",
            f"{dice_soft_mean:.6f}",
            f"{dice_hard_mean:.6f}",
            f"{iou_mean:.6f}",
            f"{reported_score:.6f}",
        ]
        if debug_mode:
            row.extend([
                f"{float(test_soft_mean):.6f}",
                f"{float(test_hard_mean):.6f}",
                f"{float(test_iou_mean):.6f}",
                f"{float(ph2_soft_mean):.6f}",
                f"{float(ph2_hard_mean):.6f}",
                f"{float(ph2_iou_mean):.6f}",
            ])
        csv_writer.writerow(row)

    if writer is not None:
        writer.add_scalar("valid/dice_soft", torch.tensor(dice_soft_mean), now_epoch)
        writer.add_scalar("valid/dice_hard", torch.tensor(dice_hard_mean), now_epoch)
        writer.add_scalar("valid/iou", torch.tensor(iou_mean), now_epoch)
        writer.add_scalar("valid/composite_score", torch.tensor(reported_score), now_epoch)

    if logger is not None:
        logger.log_metrics({
            'type': 'validation',
            'epoch': now_epoch,
            'dice_soft': float(dice_soft_mean),
            'dice_hard': float(dice_hard_mean),
            'iou': float(iou_mean),
            'composite_score': float(reported_score)
        })

    if (
        base_model_path is not None and
        topk_checkpoints is not None and
        optimizer is not None
    ):
        checkpoint_path = os.path.join(
            base_model_path,
            f"validation_epoch_{now_epoch:03d}_composite_{reported_score:.6f}.pth.tar",
        )

        should_save = False
        max_k = max(1, int(getattr(args, 'top_k', 3))) if args is not None else 3
        if len(topk_checkpoints) < max_k:
            should_save = True
        elif len(topk_checkpoints) > 0 and reported_score > topk_checkpoints[-1][2]:
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
                    'composite_score': float(reported_score),
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
                    "valid_score": float(reported_score),
                    "composite_score": float(reported_score),
                },
                checkpoint_path,
            )

            topk_checkpoints.append((now_epoch, checkpoint_path, float(reported_score)))
            topk_checkpoints.sort(key=lambda x: x[2], reverse=True)

    return iou_mean, dice_hard_mean, net_score

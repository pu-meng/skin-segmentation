import os
import csv
from typing import Dict, List, Tuple, Optional

import torch
from tqdm import tqdm

from source.tools.isic_tools.metrics import compute_sample_average_metrics


def _evaluate_dataset_metrics(
    test_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    model_path: Optional[str] = None,
) -> Tuple[float, float]:
    """
    在单个数据集上进行评估，返回 (dice_hard_mean, iou_mean)。
    评估使用 compute_sample_average_metrics（样本平均）并按样本数加权汇总。
    """
    if model_path and os.path.isfile(model_path):
        checkpoint = torch.load(model_path, weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)

    model.eval()

    dice_hard_sum = 0.0
    iou_sum = 0.0
    sample_count = 0

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Polyp Test", mininterval=0.001):
            images = images.float().cuda()
            targets = targets.float().cuda()

            outputs = model(images)
            if isinstance(outputs, (list, tuple)):
                logits = outputs[0] if len(outputs) == 3 else outputs[-1]
            else:
                logits = outputs

            if logits.dim() == 4 and logits.size(1) >= 2:
                logits = torch.softmax(logits, dim=1)
                pred_mask = torch.argmax(logits, dim=1, keepdim=True).float()
            else:
                prob = torch.sigmoid(logits)
                pred_mask = (prob > 0.5).float()

            _, batch_hard, batch_iou = compute_sample_average_metrics(pred_mask, targets)
            bsz = int(targets.size(0))
            dice_hard_sum += float(batch_hard) * bsz
            iou_sum += float(batch_iou) * bsz
            sample_count += bsz

    if sample_count > 0:
        return dice_hard_sum / sample_count, iou_sum / sample_count
    return 0.0, 0.0


def test_topk_models_polyp(
    checkpoints: List[Tuple[int, str, float]] | List[str],
    loaders: Dict[str, torch.utils.data.DataLoader],
    model: torch.nn.Module,
    results_csv: str,
) -> None:
    """
    测试Top-K（或多个）权重，输出统一的CSV：
    列为 [Epoch, Avg_Dice, Avg_IoU, CVC300_Dice, CVC300_IoU, CVCClinicDB_Dice, CVCClinicDB_IoU,
          CVCColonDB_Dice, CVCColonDB_IoU, Kvasir_Dice, Kvasir_IoU, ETISLaribPolypDB_Dice, ETISLaribPolypDB_IoU]
    其中所有Dice均为Dice_Hard；Avg_* 为五个数据集对应指标的平均。
    """
    # 规范化权重信息
    if len(checkpoints) > 0 and isinstance(checkpoints[0], tuple):
        ckpt_infos = [
            {'epoch': int(e), 'path': str(p), 'score': float(s)}
            for (e, p, s) in checkpoints  # type: ignore[misc]
        ]
    else:
        ckpt_infos = [
            {'epoch': None, 'path': str(p), 'score': None}
            for p in checkpoints  # type: ignore[misc]
        ]

    header = [
        'Epoch', 'Avg_Dice', 'Avg_IoU',
        'CVC300_Dice', 'CVC300_IoU',
        'CVCClinicDB_Dice', 'CVCClinicDB_IoU',
        'CVCColonDB_Dice', 'CVCColonDB_IoU',
        'Kvasir_Dice', 'Kvasir_IoU',
        'ETISLaribPolypDB_Dice', 'ETISLaribPolypDB_IoU'
    ]

    with open(results_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        if not ckpt_infos:
            writer.writerow({k: '' for k in header})
            return

        dataset_order = [
            'CVC300', 'CVCClinicDB', 'CVCColonDB', 'Kvasir', 'ETISLaribPolypDB'
        ]

        for info in ckpt_infos:
            path = info['path']

            per_dataset: Dict[str, Tuple[float, float]] = {}
            dice_list: List[float] = []
            iou_list: List[float] = []

            for name in dataset_order:
                loader = loaders.get(name)
                if loader is None:
                    per_dataset[name] = (0.0, 0.0)
                    dice_list.append(0.0)
                    iou_list.append(0.0)
                    continue
                dice_hard, iou = _evaluate_dataset_metrics(loader, model, model_path=path)
                per_dataset[name] = (float(dice_hard), float(iou))
                dice_list.append(float(dice_hard))
                iou_list.append(float(iou))

            avg_dice = sum(dice_list) / len(dice_list) if len(dice_list) > 0 else 0.0
            avg_iou = sum(iou_list) / len(iou_list) if len(iou_list) > 0 else 0.0

            epoch_label = f"Test Epoch {info['epoch']}" if info.get('epoch') is not None else "Test Epoch -"

            row = {
                'Epoch': epoch_label,
                'Avg_Dice': f"{avg_dice:.6f}",
                'Avg_IoU': f"{avg_iou:.6f}",
                'CVC300_Dice': f"{per_dataset['CVC300'][0]:.6f}",
                'CVC300_IoU': f"{per_dataset['CVC300'][1]:.6f}",
                'CVCClinicDB_Dice': f"{per_dataset['CVCClinicDB'][0]:.6f}",
                'CVCClinicDB_IoU': f"{per_dataset['CVCClinicDB'][1]:.6f}",
                'CVCColonDB_Dice': f"{per_dataset['CVCColonDB'][0]:.6f}",
                'CVCColonDB_IoU': f"{per_dataset['CVCColonDB'][1]:.6f}",
                'Kvasir_Dice': f"{per_dataset['Kvasir'][0]:.6f}",
                'Kvasir_IoU': f"{per_dataset['Kvasir'][1]:.6f}",
                'ETISLaribPolypDB_Dice': f"{per_dataset['ETISLaribPolypDB'][0]:.6f}",
                'ETISLaribPolypDB_IoU': f"{per_dataset['ETISLaribPolypDB'][1]:.6f}",
            }

            writer.writerow(row)

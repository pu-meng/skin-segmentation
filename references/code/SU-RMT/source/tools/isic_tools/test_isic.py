import os
import csv
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import torch
from tqdm import tqdm

from source.tools.isic_tools.metrics import compute_batch_avearge_metrics


def test_topk_models_isic(
    checkpoints: List[Tuple[int, str, float]] | List[str],
    test_loader: torch.utils.data.DataLoader,
    ph2_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    results_csv: str,
) -> None:
    """
    Test Top-K (or several) models and write the final summary CSV (contains Epoch text and three metrics; no accuracy).
    Checkpoints can be a list of (epoch, path, score) tuples, or just a list of paths.

    Args:
        checkpoints (List[Tuple[int, str, float]] | List[str]): List of (epoch, path, score) tuples or just a list of paths
        test_loader (torch.utils.data.DataLoader): Test data loader
        ph2_loader (torch.utils.data.DataLoader): PH2 data loader
        model (torch.nn.Module): Model to test
        results_csv (str): Path to save the results CSV

    Returns:
        None
    """
    # Normalize to list with metadata (keep epoch/score for later use if needed)
    if len(checkpoints) > 0 and isinstance(checkpoints[0], tuple):
        ckpt_infos = [
            {
                'epoch': int(e),
                'path': str(p),
                'score': float(s),
            }
            for (e, p, s) in checkpoints  # type: ignore[misc]
        ]
    else:
        ckpt_infos = [
            {
                'epoch': None,
                'path': str(p),
                'score': None,
            }
            for p in checkpoints  # type: ignore[misc]
        ]

    if not ckpt_infos:
        # If no available weights, return empty header
        with open(results_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'Epoch',
                'Test_SoftDice', 'Test_HardDice', 'Test_IoU',
                'PH2_SoftDice', 'PH2_HardDice', 'PH2_IoU'
            ])
            writer.writeheader()
        return

    rows_all: list[dict] = []

    for _, info in enumerate(ckpt_infos):
        path = info['path']
        # ISIC Test
        isic_results = test_isic(
            test_loader=test_loader,
            model=model,
            model_path=path,
        )
        # PH2 Test
        ph2_results = test_isic(
            test_loader=ph2_loader,
            model=model,
            model_path=path,
        )

        # Parse numerical values
        def _mean_from(res: dict, key: str) -> float:
            val = res.get(key, (0.0, 0.0))
            return float(val[0]) if isinstance(val, tuple) else float(val)

        test_soft = _mean_from(isic_results, 'dice_soft')
        test_hard = _mean_from(isic_results, 'dice_hard')
        test_iou = _mean_from(isic_results, 'iou')
        ph2_soft = _mean_from(ph2_results, 'dice_soft')
        ph2_hard = _mean_from(ph2_results, 'dice_hard')
        ph2_iou = _mean_from(ph2_results, 'iou')

        epoch_label = f"Test Epoch {info['epoch']}" if info.get('epoch') is not None else "Test Epoch -"

        row = {
            'Epoch': epoch_label,
            'Test_SoftDice': f"{test_soft:.6f}",
            'Test_HardDice': f"{test_hard:.6f}",
            'Test_IoU': f"{test_iou:.6f}",
            'PH2_SoftDice': f"{ph2_soft:.6f}",
            'PH2_HardDice': f"{ph2_hard:.6f}",
            'PH2_IoU': f"{ph2_iou:.6f}",
        }

        rows_all.append(row)

    # Write multiple rows: write each checkpoint result row by row into the same CSV
    header = [
        'Epoch',
        'Test_SoftDice', 'Test_HardDice', 'Test_IoU',
        'PH2_SoftDice', 'PH2_HardDice', 'PH2_IoU'
    ]
    with open(results_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        if len(rows_all) == 0:
            writer.writerow({k: '' for k in header})
        else:
            for r in rows_all:
                # Only write header
                writer.writerow({k: r.get(k, '') for k in header})


def test_isic(
    test_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    model_path: Optional[str],
) -> Dict[str, Union[Tuple[float, float], float]]:
    """
    Test the model on the test dataset and calculate multiple evaluation metrics.

    Args:
        test_loader (torch.utils.data.DataLoader): Test data loader
        model (torch.nn.Module): Model to test
        model_path (Optional[str]): Optional model weight path (use current model parameters if None)

    Returns:
        Dictionary containing all evaluation metrics, each metric contains (mean, std)
        The dictionary contains:
            - 'dice_soft': Soft Dice
            - 'dice_hard': Hard Dice
            - 'iou': IoU
            - 'dice_soft_std': Soft Dice standard deviation
            - 'dice_hard_std': Hard Dice standard deviation
            - 'iou_std': IoU standard deviation
    """

    # Simple metric collection container
    dice_soft_list: List[float] = []
    dice_hard_list: List[float] = []
    iou_list: List[float] = []

    # Load specified model weights (optional)
    if model_path and os.path.isfile(model_path):
        checkpoint = torch.load(model_path, weights_only=False)
        model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=False)
        if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
            print(f"=> Successfully loaded (epoch {checkpoint['epoch']})")

    # Set to evaluation mode
    model.eval()

    # Start testing (no longer write batch-by-batch text logs)
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Test"):
            images = images.float().cuda()
            targets = targets.float().cuda()

            # Forward inference
            outputs = model(images)

            # C>=2: argmax directly (equivalent to softmax+argmax); C==1: sigmoid+0.5
            if outputs.dim() == 4 and outputs.size(1) >= 2:
                outputs = torch.softmax(outputs, dim=1)
                output_dis = torch.argmax(outputs, dim=1, keepdim=True)
            else:
                prob = torch.sigmoid(outputs)
                output_dis = (prob > 0.5).long()

            # Calculate and collect all metrics
            batch_dice_soft, batch_dice_hard, batch_iou = compute_batch_avearge_metrics(output_dis, targets)
            dice_soft_list.append(batch_dice_soft)
            dice_hard_list.append(batch_dice_hard)
            iou_list.append(batch_iou)

    # Calculate and organize results
    results: Dict[str, Union[Tuple[float, float], float]] = {}
    if len(dice_soft_list) > 0:
        results['dice_soft'] = (float(np.mean(dice_soft_list)), float(np.std(dice_soft_list)))
        results['dice_hard'] = (float(np.mean(dice_hard_list)), float(np.std(dice_hard_list)))
        results['iou'] = (float(np.mean(iou_list)), float(np.std(iou_list)))
    else:
        results['dice_soft'] = (0.0, 0.0)
        results['dice_hard'] = (0.0, 0.0)
        results['iou'] = (0.0, 0.0)

    return results

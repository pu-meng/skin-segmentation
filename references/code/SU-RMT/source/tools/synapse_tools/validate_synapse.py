import os
import csv
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np
from tqdm import tqdm

import distorch
from scipy.ndimage import zoom
from torch.utils.tensorboard import SummaryWriter

from source.utils.log import Logger
from source.config import TrainConfig


def validate_synapse(
    val_loader: torch.utils.data.DataLoader,
    eval_model: torch.nn.Module,
    orig_model: torch.nn.Module,
    ema_model: Optional[torch.nn.Module],
    args: TrainConfig,
    epoch: int,
    writer: Optional[SummaryWriter],
    logger: Optional[Logger],
    base_model_path: str,
    optimizer: torch.optim.Optimizer,
    validation_results_csv: str,
    topk_checkpoints: List[Tuple[int, str, float]],
    num_classes: int = 9
) -> Dict[str, float]:
    """
    Wrapper for validation, logging, CSV writing, and Top-K weight management.
    Args:
        val_loader(torch.utils.data.DataLoader): Validation data loader
        eval_model(torch.nn.Module): Evaluation model
        orig_model(torch.nn.Module): Original model
        ema_model(torch.nn.Module): EMA model
        args(TrainConfig): Training arguments
        epoch(int): Current epoch
        writer(Optional[SummaryWriter]): TensorBoard writer
        logger(Optional[Logger]): Logger
        base_model_path(str): Base model path
        optimizer(torch.optim.Optimizer): Optimizer
        validation_results_csv(str): Validation results CSV file path
        topk_checkpoints(List[Tuple[int, str, float]]): Top-K checkpoints
        num_classes(int): Number of classes
    Returns:
        validation_results(Dict[str, float]): Validation results dictionary
        The dictionary contains:
            - 'dice': Average Dice
            - 'hd95': Average HD95
            - 'jaccard': Average Jaccard
            - 'composite_score': Composite score
            - 'case_count': Number of valid validation cases
            - 'metric_list': Detailed per-class metrics

    Note: This function will update the topk_checkpoints list in place.
    """
    patch_size = args.out_size[0]
    # Run fast validation
    validation_results = _validate_synapse(
        val_loader=val_loader,
        model=eval_model,
        num_classes=num_classes,
        patch_size=patch_size,
    )

    valid_dc = validation_results['dice']
    valid_hd95 = validation_results['hd95']
    composite_score = validation_results['composite_score']
    valid_net_score = composite_score

    # Structured logging
    if logger is not None:
        val_metrics_log = {
            'type': 'validation',
            'epoch': epoch,
            'dice': validation_results['dice'],
            'hd95': validation_results['hd95'],
            'jaccard': validation_results['jaccard'],
            'composite_score': validation_results['composite_score']
        }
        logger.log_metrics(val_metrics_log)

    # Write to CSV
    csv_file_path = validation_results_csv
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, 'w', newline='', encoding="utf-8") as csvfile:
            csv_writer = csv.writer(csvfile)
            headers = ['Epoch', 'Avg_Dice', 'Avg_HD95', 'Composite_Score']
            for i in range(1, num_classes):
                headers.extend([f'Class{i}_Dice', f'Class{i}_HD95'])
            csv_writer.writerow(headers)

    row_data = [f"Validation Epoch {epoch}", f"{valid_dc:.6f}", f"{valid_hd95:.6f}", f"{composite_score:.6f}"]
    if 'metric_list' in validation_results and len(validation_results['metric_list']) > 0:
        metric_list = validation_results['metric_list']
        for i in range(1, num_classes):
            if i-1 < len(metric_list):
                class_dice = metric_list[i-1][0]
                class_hd95 = metric_list[i-1][1]
                row_data.extend([f"{class_dice:.6f}", f"{class_hd95:.6f}"])
            else:
                row_data.extend(["0.000000", "100.000000"])
    else:
        for _ in range(1, num_classes):
            row_data.extend(["0.000000", "100.000000"])

    with open(csv_file_path, 'a', newline='', encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(row_data)

    # Write to TensorBoard
    if writer is not None:
        writer.add_scalar("valid/average_dice", torch.tensor(valid_dc), epoch)
        writer.add_scalar("valid/average_hd95", torch.tensor(valid_hd95), epoch)
        writer.add_scalar("valid/composite_score", torch.tensor(composite_score), epoch)
        writer.add_scalar("valid/jaccard", torch.tensor(validation_results['jaccard']), epoch)

    # Dynamic management of Top-K best weights
    checkpoint_path = os.path.join(
        base_model_path,
        f"validation_epoch_{epoch:03d}_composite_{valid_net_score:.6f}.pth.tar",
    )

    should_save = False
    if len(topk_checkpoints) < max(1, int(args.top_k)):
        should_save = True
    elif len(topk_checkpoints) > 0 and valid_net_score > topk_checkpoints[-1][2]:
        worst_checkpoint = topk_checkpoints[-1]
        if os.path.exists(worst_checkpoint[1]):
            os.remove(worst_checkpoint[1])
        topk_checkpoints.pop()
        should_save = True

    if should_save:
        if logger is not None:
            logger.log_metrics({
                'type': 'checkpoint_saved',
                'epoch': epoch,
                'checkpoint': os.path.basename(checkpoint_path),
                'composite_score': valid_net_score
            })

        if getattr(args, 'use_ema', False) and ema_model is not None:
            state_dict = ema_model.module.state_dict()
        else:
            state_dict = orig_model.state_dict()

        torch.save({
            "epoch": epoch,
            "state_dict": state_dict,
            "opt_dict": optimizer.state_dict(),
            "composite_score": valid_net_score,
        }, checkpoint_path)

        topk_checkpoints.append((epoch, checkpoint_path, valid_net_score))
        topk_checkpoints.sort(key=lambda x: x[2], reverse=True)

    return validation_results


def _validate_synapse(
    val_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    num_classes: int = 9,
    max_cases: int = 12,  # 限制验证病例数以加速
    patch_size: int = 224  # 支持自定义分辨率
) -> Dict[str, float]:
    """
    Fast Synapse validation function: full 3D volume processing based on test_single_volume
    Use distorch for GPU-accelerated 3D HD95 calculations, return results in the same format as test_synapse

    Args:
        val_loader (torch.utils.data.DataLoader): Validation data loader
        model (torch.nn.Module): Model
        num_classes (int): Number of classes
        max_cases (int): Maximum number of validation cases (for speed)
        patch_size (int): Input patch size

    Returns:
        results (Dict[str, float]): Validation metrics dictionary, compatible with test_synapse, containing 3D HD95 metrics
        The dictionary contains:
            - 'dice': Average Dice
            - 'hd95': Average HD95
            - 'jaccard': Average Jaccard
            - 'case_count': Number of valid validation cases
            - 'metric_list': Detailed per-class metrics
            - 'composite_score': Composite score
    """
    model.eval()

    # Accumulated metric list - each element is an array of (dice, hd95, jaccard) for a case
    metric_list = []
    case_count = 0

    # Create progress bar
    pbar = tqdm(total=max_cases, desc="Validation progress", ncols=120, leave=False)

    with torch.no_grad():
        for i_batch, batch in enumerate(val_loader):
            if case_count >= max_cases:
                break

            try:
                if isinstance(batch, dict):
                    h, w = batch["image"].size()[2:]  # Get image size
                    image = batch["image"].float().cuda()   # [B, C, D, H, W] or [B, D, H, W]
                    label = batch["label"].long().cuda()    # [B, D, H, W]
                    case_name = batch.get('case_name', [f'case_{i_batch}'])[0]
                else:
                    continue

                # 将tensor转换为numpy数组，并移除批次维度
                # Tensor -> numpy array, (B, C, D, H, W) -> (C, D, H, W)
                image_np = image.squeeze(0).cpu().detach().numpy()
                label_np = label.squeeze(0).cpu().detach().numpy()

                # If image has channel dimension, take the first channel
                if len(image_np.shape) == 4 and image_np.shape[0] <= 3:  # [C, D, H, W]
                    image_np = image_np[0]  # [D, H, W]

                # Ensure we have a 3D volume [D, H, W]
                if len(image_np.shape) != 3 or len(label_np.shape) != 3:
                    continue

                # Initialize prediction array, same size as label
                prediction = np.zeros_like(label_np)

                # Process each slice - same as test_single_volume
                for ind in range(image_np.shape[0]):
                    # Extract single slice
                    slice_img = image_np[ind, :, :]
                    x, y = slice_img.shape[0], slice_img.shape[1]

                    # If slice size is different from patch_size, resize
                    if x != patch_size or y != patch_size:
                        slice_img = zoom(slice_img, (patch_size / x, patch_size / y), order=3)  # cubic interpolation

                    # Convert to tensor and add batch and channel dimensions
                    # [D, H, W] -> [1, 1, D, H, W]
                    input_tensor = torch.from_numpy(slice_img).unsqueeze(0).unsqueeze(0).float().cuda()

                    # Model inference
                    outputs = model(input_tensor)

                    # Apply softmax and get prediction class
                    output_softmax = torch.softmax(outputs, dim=1)

                    out = torch.argmax(output_softmax, dim=1).squeeze(0)
                    out = out.cpu().detach().numpy()

                    # If slice size is different from patch_size, resize
                    if x != patch_size or y != patch_size:
                        out = zoom(out, (x / patch_size, y / patch_size), order=0)  # nearest neighbor interpolation

                    # Store the prediction of this slice
                    prediction[ind] = out

                # Calculate metrics for each class - GPU-accelerated version, avoid CPU-GPU data transfer
                prediction_tensor = torch.from_numpy(prediction).cuda()
                label_tensor = torch.from_numpy(label_np).cuda()

                metric_i = []
                for i in range(1, num_classes):  # Skip background class
                    # Extract prediction and ground truth for this class on GPU
                    pred_c_tensor = (prediction_tensor == i)
                    target_c_tensor = (label_tensor == i)

                    # Calculate foreground pixel count on GPU
                    pred_sum = pred_c_tensor.sum().float()
                    target_sum = target_c_tensor.sum().float()

                    # Calculate metrics - all done on GPU
                    if pred_sum > 0 and target_sum > 0:
                        # Both have foreground pixels
                        # Dice calculation on GPU - convert bool tensor to float
                        pred_c_float = pred_c_tensor.float()
                        target_c_float = target_c_tensor.float()
                        intersection = (pred_c_float * target_c_float).sum()
                        dice = (2.0 * intersection) / (pred_sum + target_sum + 1e-8)

                        # Jaccard calculation on GPU
                        union = pred_sum + target_sum - intersection
                        jaccard = intersection / (union + 1e-8) if union > 0 else 0.0

                        # Convert to float for subsequent calculations
                        dice = float(dice.cpu())
                        jaccard = float(jaccard.cpu())

                        # Use distorch to calculate real 3D HD95 (GPU-accelerated) - reference medpy's hd95 implementation
                        try:
                            # Use GPU tensor directly, convert to bool type for HD95 calculation
                            pred_tensor = pred_c_tensor.bool()
                            target_tensor = target_c_tensor.bool()

                            # distorch requires 3D volume to be at least 4D tensor [B, D, H, W]
                            # Add batch dimension to both tensors
                            if pred_tensor.dim() == 3:  # [D, H, W]
                                pred_tensor = pred_tensor.unsqueeze(0)  # [1, D, H, W]
                                target_tensor = target_tensor.unsqueeze(0)  # [1, D, H, W]

                                # Final shape check
                                assert pred_tensor.shape == target_tensor.shape, f"Final tensor shape mismatch: {pred_tensor.shape} vs {target_tensor.shape}" # noqa

                                # Calculate 3D boundary distance metrics
                                metrics = distorch.boundary_metrics(pred_tensor, target_tensor)

                                # Use symmetric HD95 according to medpy's hd95 implementation
                                # medpy calculation: numpy.percentile(numpy.hstack((hd1, hd2)), 95)
                                # We use symmetric HD95 results provided by distorch, ensure return scalar
                                hd95_1_to_2 = float(metrics.Hausdorff95_1_to_2.cpu().item())
                                hd95_2_to_1 = float(metrics.Hausdorff95_2_to_1.cpu().item())
                                hd95 = max(hd95_1_to_2, hd95_2_to_1)

                        except Exception as e:
                            # If calculation fails, use default value and output detailed error information
                            print(f"distorch HD95 calculation failed (class {i}): {str(e)}")
                            print(f"  pred_tensor shape: {pred_c_tensor.shape}, target_tensor shape: {target_c_tensor.shape}")
                            hd95 = 0.0
                    elif pred_sum > 0 and target_sum == 0:
                        # False positive (predicted but not actual)
                        dice, hd95, jaccard = 0.0, 0.0, 0.0
                    else:
                        # Both have no foreground pixels
                        dice, hd95, jaccard = 0.0, 0.0, 0.0

                    # Ensure all metrics are scalar values
                    dice = float(dice) if not isinstance(dice, (int, float)) else dice
                    hd95 = float(hd95) if not isinstance(hd95, (int, float)) else hd95
                    jaccard = float(jaccard) if not isinstance(jaccard, (int, float)) else jaccard

                    # Add to metric list: (dice, hd95, jaccard)
                    metric_i.append((dice, hd95, jaccard))

                # Accumulate metrics - same as test_synapse, ensure correct shape
                try:
                    # Check type of each element in metric_i
                    for idx, m in enumerate(metric_i):
                        if not isinstance(m, tuple) or len(m) != 3:
                            print(f"  Error: case {case_name} class {idx+1} metric format error: {type(m)}, value: {m}")
                            raise ValueError(f"Metric format error: {m}")
                        for j, val in enumerate(m):
                            if not isinstance(val, (int, float)):
                                print(f"  Error: case {case_name} class {idx+1} metric value {j} is not a number: {type(val)}, value: {val}") # noqa
                                raise ValueError(f"Metric value is not a number: {val}")

                    metric_i_array = np.array(metric_i)
                    expected_shape = (num_classes - 1, 3)  # (8, 3)

                    if metric_i_array.shape == expected_shape:
                        metric_list.append(metric_i_array)
                        case_count += 1

                        # Update progress bar
                        mean_dice = np.mean([m[0] for m in metric_i])
                        mean_hd95 = np.mean([m[1] for m in metric_i])
                        mean_jaccard = np.mean([m[2] for m in metric_i])
                        pbar.set_postfix({
                            'case': case_name,
                            'dice': f'{mean_dice:.4f}',
                            'hd95': f'{mean_hd95:.2f}',
                            'jaccard': f'{mean_jaccard:.4f}'
                        })
                        pbar.update(1)
                    else:
                        print(f"  Warning: case {case_name} metric shape abnormal: {metric_i_array.shape}, expected: {expected_shape}") # noqa
                        # Skip cases with abnormal shape
                        continue

                except Exception as array_error:
                    print(f"  Error: case {case_name} array conversion failed: {str(array_error)}")
                    print(f"  metric_i length: {len(metric_i)}")
                    for idx, m in enumerate(metric_i):
                        print(f"    class {idx+1}: {type(m)}, {m}")
                    continue

            except Exception as e:
                print(f"  Error: case {i_batch} validation failed: {str(e)}")
                continue

    # Calculate overall statistics - same as test_synapse
    if case_count > 0:
        # Safe array conversion and average calculation, avoid shape mismatch errors
        # First check if all metric arrays have the same shape
        expected_shape = (num_classes - 1, 3)  # 8 foreground classes, 3 metrics
        valid_metrics = []

        for i, metric in enumerate(metric_list):
            if isinstance(metric, np.ndarray) and metric.shape == expected_shape:
                valid_metrics.append(metric)
            else:
                print(f"  Warning: case {i} metric shape abnormal: {metric.shape if isinstance(metric, np.ndarray) else type(metric)}") # noqa

        if valid_metrics:
            # Convert metrics with consistent shape to numpy array
            metric_array = np.stack(valid_metrics, axis=0)  # shape: (valid_case_count, num_classes-1, 3)
            metric_list_mean = np.mean(metric_array, axis=0)  # shape: (num_classes-1, 3)

            # Calculate average values for all classes
            performance = np.mean(metric_list_mean, axis=0)[0]  # Average Dice
            mean_hd95 = np.mean(metric_list_mean, axis=0)[1]  # Average HD95
            mean_jaccard = np.mean(metric_list_mean, axis=0)[2]  # Average Jaccard

            # Calculate composite score
            composite_score = _compute_synapse_score(performance, mean_hd95)
        else:
            # If no valid metrics, use default values
            print("  Warning: no valid metrics data")
            performance = 0.0
            mean_hd95 = 100.0
            mean_jaccard = 0.0
            metric_list_mean = np.zeros((num_classes - 1, 3))

        results = {
            'dice': performance,
            'hd95': mean_hd95,
            'jaccard': mean_jaccard,
            'composite_score': composite_score,
            'case_count': case_count,
            'metric_list': metric_list_mean  # Add detailed per-class metrics
        }

        # Close progress bar
        pbar.close()
    else:
        # Close progress bar
        pbar.close()
        results = {
            'dice': 0.0,
            'hd95': 100.0,
            'jaccard': 0.0,
            'case_count': 0,
            'metric_list': []
        }
    return results


def _compute_synapse_score(
    dice: float, hd95: float, dice_weight: float = 0.9, hd95_weight: float = 0.1
) -> float:
    """
    Compute composite score, balance Dice and HD95 metrics.

    Args:
        dice (float): Dice coefficient (Higher is better)
        hd95 (float): 95% Hausdorff distance (Lower is better)
        dice_weight (float): Dice weight (Default to 0.9)
        hd95_weight (float): HD95 weight (Default to 0.1)

    Returns:
        composite_score (float): Composite score (0-1, Higher is better)
    """
    # Normalize Dice: assume good Dice > 0.8
    normalized_dice = min(dice / 0.9, 1.0)

    # Normalize HD95: assume good HD95 < 25.0
    normalized_hd95 = max(0, min(1, (50.0 - hd95) / 50.0))  # HD95=0 -> 1, HD95=50 -> 0

    # Compute weighted composite score
    composite_score = dice_weight * normalized_dice + hd95_weight * normalized_hd95

    return composite_score

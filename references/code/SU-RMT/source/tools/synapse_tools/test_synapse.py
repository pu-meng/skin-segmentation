import csv
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from scipy.ndimage import zoom
from medpy import metric

from source.utils.log import Logger


def test_topk_models(
    checkpoints: List[Tuple[int, str, float]],
    test_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    results_csv: str,
    logger: Logger,
    num_classes: int = 9,
    patch_size: Tuple[int, int] = (224, 224)
) -> None:
    """
    Test the top K best models and output the complete CSV results.

    Args:
        checkpoints: List of tuples containing (epoch, checkpoint_path, validation_score)
        test_loader: Test data loader
        model: Model to test
        results_csv: Path to save the results
        logger: Logger object
        num_classes: Number of classes

    Returns:
        None
    """
    # If no checkpoints are available, log an error and return
    if not checkpoints:
        logger.log_metrics({
            'type': 'error',
            'message': 'No validation checkpoints available for testing'
        })
        return

    # Record the final model selection
    for i, (epoch, path, score) in enumerate(checkpoints):
        logger.log_metrics({
            'type': 'final_model_selection',
            'rank': i+1,
            'epoch': epoch,
            'composite_score': score,
            'checkpoint': path.split('/')[-1]
        })

    final_results_path = results_csv

    # Create the CSV file and write the headers
    with open(final_results_path, 'w', newline='', encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write the headers
        headers = ['Epoch', 'Avg_Dice', 'Avg_HD95']
        for i in range(1, 9):  # 8 classes
            headers.extend([f'Class{i}_Dice', f'Class{i}_HD95'])
        csv_writer.writerow(headers)

    # Test each model
    for i, (epoch, checkpoint_path, validation_score) in enumerate(checkpoints):
        # Test the model
        logger.log_metrics({
            'type': 'test_start',
            'model_rank': i+1,
            'epoch': epoch,
            'validation_score': validation_score
        })

        # Use test_synapse for complete testing
        results = _test_synapse(
            test_loader=test_loader,
            model=model,
            model_path=checkpoint_path,
            date_type=f"Test Epoch {epoch}",
            num_classes=num_classes,
            patch_size=patch_size
        )

        avg_dice = results['dice'][0]  # Average Dice
        avg_hd95 = results['hd95'][0]  # Average HD95

        # Record the test results
        logger.log_metrics({
            'type': 'test_result',
            'model_rank': i+1,
            'epoch': epoch,
            'avg_dice': avg_dice,
            'avg_hd95': avg_hd95
        })

        # Prepare the CSV row data
        row_data = [f"Test Epoch {epoch}", f"{avg_dice:.6f}", f"{avg_hd95:.6f}"]

        # Add the results for each class
        for class_idx in range(1, num_classes):  # 1-8 classes
            class_dice = results[f'dice_class_{class_idx}'][0]
            class_hd95 = results[f'hd95_class_{class_idx}'][0]
            row_data.extend([f"{class_dice:.6f}", f"{class_hd95:.6f}"])

        # Write to the CSV file
        with open(final_results_path, 'a', newline='', encoding="utf-8") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(row_data)


def _test_synapse(
    test_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    model_path: str,
    date_type: str,
    num_classes: int = 9,
    patch_size: Tuple[int, int] = (224, 224)
) -> Dict[str, Tuple[float, float]]:
    """
    Test Synapse dataset with given model and calculate multiple evaluation metrics.

    Args:
        test_loader (torch.utils.data.DataLoader): Test data loader
        model (torch.nn.Module): PyTorch model to evaluate
        model_path (str): Model save path
        date_type (str): Dataset type name
        num_classes (int): Number of segmentation classes, default 9

    Returns:
        results (Dict[str, Tuple[float, float]]):Dictionary containing all evaluation metrics, each metric contains (mean, std)
        - dice: (mean, std)
        - hd95: (mean, std)
        - dice_class_i: (mean, std) (i = 1, 2, ..., 8)
        - hd95_class_i: (mean, std) (i = 1, 2, ..., 8)
    """
    print(f"================ {date_type} Testing Start ================")
    print(f"Load model from {model_path}")
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"=> Load model from {model_path} (epoch {checkpoint.get('epoch', 'unknown')})")

    # Set to evaluation mode
    model.eval()

    # Accumulate per-class metrics across cases: shape [(num_classes-1), 2] -> (dice, hd95)
    class_metrics_sum = np.zeros((num_classes - 1, 2), dtype=float)
    for i_batch, batch in enumerate(tqdm(test_loader, desc=f"{date_type} Testing")):

        image = batch["image"].float().cuda()
        label = batch["label"].long().cuda()
        case_name = batch.get('case_name', [f'case_{i_batch}'])[0]

        metrics_per_case = _test_single_volume(
            image=image,
            label=label,
            model=model,
            classes=num_classes,
            patch_size=patch_size,
        )

        class_metrics_sum += np.array(metrics_per_case)

        case_mean_dice = float(np.mean([m[0] for m in metrics_per_case])) if metrics_per_case else 0.0
        case_mean_hd95 = float(np.mean([m[1] for m in metrics_per_case])) if metrics_per_case else 0.0
        print(f"Case {case_name}: Mean Dice {case_mean_dice:.4f} | Mean HD95 {case_mean_hd95:.4f}")

    # Calculate average metrics (divide by number of samples)
    class_metrics_mean = class_metrics_sum / len(test_loader)

    # Dataset-level mean across classes
    dataset_mean_dice = float(np.mean(class_metrics_mean[:, 0]))
    dataset_mean_hd95 = float(np.mean(class_metrics_mean[:, 1]))

    # Only print overall metrics
    print(f"Test results: Average Dice: {dataset_mean_dice:.4f}, Average HD95: {dataset_mean_hd95:.4f}")

    # Prepare results dictionary, compatible with original code
    results: Dict[str, Tuple[float, float]] = {}
    results['dice'] = (dataset_mean_dice, float(np.std(class_metrics_mean[:, 0])))
    results['hd95'] = (dataset_mean_hd95, float(np.std(class_metrics_mean[:, 1])))

    # Add metrics for each class
    for i in range(1, num_classes):
        results[f'dice_class_{i}'] = (float(class_metrics_mean[i-1][0]), 0.0)
        results[f'hd95_class_{i}'] = (float(class_metrics_mean[i-1][1]), 0.0)
    return results


def _test_single_volume(
    image: torch.Tensor,
    label: torch.Tensor,
    model: torch.nn.Module,
    classes: int,
    patch_size: Tuple[int, int] = (224, 224),
) -> List[Tuple[float, float]]:
    """
    Test a single 3D volume.

    Args:
        image (torch.Tensor): Input image, shape [B, D, H, W]
        label (torch.Tensor): Label, shape [B, D, H, W]
        model (torch.nn.Module): Model
        classes (int): Number of classes
        patch_size (Tuple[int, int]): Input patch size [H, W]

    Returns:
        List[Tuple[float, float]]: (dice, hd95) for each class
    """
    # torch.Tensor to np.ndarray (B, D, H, W) -> (D, H, W)
    image = image.squeeze(0).cpu().detach().numpy()
    label = label.squeeze(0).cpu().detach().numpy()

    # Initialize prediction array, (D, H, W)
    prediction = np.zeros_like(label)

    # Process each slice (D, H, W)
    print(f"Processing 3D volume: {image.shape[0]} slices")
    for ind in range(image.shape[0]):
        # (D, H, W) -> (H, W)
        slice_img = image[ind, :, :]

        # Get slice size
        x, y = slice_img.shape[0], slice_img.shape[1]

        # (H, W) -> (patch_size[0], patch_size[1])
        if x != patch_size[0] or y != patch_size[1]:
            slice_img = zoom(slice_img, (patch_size[0] / x, patch_size[1] / y), order=3)  # Cubic spline interpolation

        # np.ndarray to torch.Tensor (patch_size[0], patch_size[1]) -> (1, 1, patch_size[0], patch_size[1])
        input_tensor = torch.from_numpy(slice_img).unsqueeze(0).unsqueeze(0).float().cuda()

        # Model inference (1, 1, patch_size[0], patch_size[1]) -> (1, 1, patch_size[0], patch_size[1])
        model.eval()
        with torch.no_grad():
            outputs = model(input_tensor)

            # Apply softmax and get prediction class
            output_softmax = torch.softmax(outputs, dim=1)
            out = torch.argmax(output_softmax, dim=1).squeeze(0)
            out = out.cpu().detach().numpy()

            # (patch_size[0], patch_size[1]) -> (H, W)
            if x != patch_size[0] or y != patch_size[1]:
                out = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)  # Nearest neighbor interpolation

            # Store the prediction of this slice (D, H, W)
            prediction[ind] = out

    # Calculate metrics for each class
    metric_list = []
    for i in range(1, classes):  # Skip background class
        metric_list.append(_calculate_metric_percase(prediction == i, label == i))

    return metric_list


def _calculate_metric_percase(
    prediction: np.ndarray, ground_truth: np.ndarray
) -> Tuple[float, float]:
    """
    Compute the Dice and HD95 metrics for a single case.

    Args:
        prediction: Predicted mask (binarized), shape [D, H, W]
        ground_truth: Ground truth mask (binarized), shape [D, H, W]

    Returns:
        Tuple[float, float]: (dice, hd95)
    """
    # Binarize the prediction and ground truth
    prediction = np.copy(prediction)
    ground_truth = np.copy(ground_truth)
    prediction[prediction > 0] = 1
    ground_truth[ground_truth > 0] = 1

    pred_sum = int(np.sum(prediction))
    gt_sum = int(np.sum(ground_truth))

    # Handle empty masks to avoid medpy errors
    if gt_sum == 0 and pred_sum == 0:
        # Both empty: perfect agreement for this class
        return 1.0, 0.0
    if gt_sum == 0 and pred_sum > 0:
        # No GT but predicted something: penalize
        return 0.0, 100.0
    if gt_sum > 0 and pred_sum == 0:
        # GT exists but prediction empty: penalize
        return 0.0, 100.0

    dice = metric.binary.dc(prediction, ground_truth)
    hd95 = metric.binary.hd95(prediction, ground_truth)
    return dice, hd95

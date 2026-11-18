import torch
from torch import nn


class DiceLoss(nn.Module):
    """
    Standard multi-class Dice loss, calculated per channel then averaged.

    Args:
        include_background (bool):
         - Whether to include background class (channel 0) in loss.
         - Default is False, i.e., ignore background.
        smooth (float):
         - Smoothing constant to avoid division by zero.
    """
    def __init__(self, include_background: bool = False, smooth: float = 1e-5):
        super().__init__()
        self.include_background = include_background
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Dice loss.

        Args:
            pred: Model logits or probabilities with shape (N, C, H, W).
            target: Ground truth with shape (N, C, H, W), typically one-hot or soft labels.

        Returns:
            Scalar Dice loss.
        """
        num_classes = pred.shape[1]
        pred_prob = torch.softmax(pred, dim=1)

        # Compute per-channel Dice over batch and spatial dims
        dims = (0, 2, 3)
        intersection = (pred_prob * target).sum(dim=dims)
        pred_sum = pred_prob.sum(dim=dims)
        target_sum = target.sum(dim=dims)
        dice_per_channel = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)

        # Optionally exclude background (channel 0) when more than 1 class exists
        if not self.include_background and num_classes > 1:
            dice_per_channel = dice_per_channel[1:]

        loss_per_channel = 1.0 - dice_per_channel
        return loss_per_channel.mean()

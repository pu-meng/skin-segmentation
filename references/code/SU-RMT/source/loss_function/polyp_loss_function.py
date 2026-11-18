from typing import Dict, Any

import torch
from torch import nn
import torch.nn.functional as F

from .dice_loss import DiceLoss


class PolypLossFunction(nn.Module):
    """
    Polyp segmentation loss function.
    """
    def __init__(
        self,
        num_classes: int,
        dice_weight: float = 0.7,
        ce_weight: float = 0.3,
        loss_strategy: str = "Normal",
    ):
        """
        Initialize Polyp loss function.

        Args:
            num_classes: Number of classes.
            dice_weight: Dice weight.
            ce_weight: Cross entropy weight.
            loss_strategy: Loss strategy (for future extensions).
        """
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.loss_strategy = loss_strategy
        self.current_epoch = 0

        self.dice_loss = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, Any]:
        """
        Calculate composite loss.
        
        Args:
            pred: Model predictions (logits), shape (N, C, H, W)
            target: Ground truth labels, shape (N, H, W) or (N, 1, H, W)
        
        Returns:
            Dictionary containing total_loss, dice_loss, ce_loss, and weights
        """
        # Handle different target shapes
        if target.dim() == 3:
            n, h, w = target.shape
        elif target.dim() == 4 and target.size(1) == 1:
            # Compatible with (N,1,H,W)
            n, _, h, w = target.shape
            target = target.squeeze(1)
        else:
            raise ValueError(f"Unexpected target shape for one-hot: {target.shape}")

        # Create one-hot encoding
        target_one_hot = torch.zeros(
            (n, self.num_classes, h, w),
            device=target.device,
            dtype=torch.float32,
        )
        target_one_hot.scatter_(1, target.long().unsqueeze(1), 1.0)

        # Calculate Dice loss
        dice_loss_val = self.dice_loss(pred, target_one_hot)

        # Calculate Cross-Entropy loss
        # CE = - sum_c (y_onehot * log_softmax(pred)), then average over (N,H,W)
        log_probs = F.log_softmax(pred, dim=1)
        ce_per_pixel = -(target_one_hot * log_probs).sum(dim=1)
        ce_loss_val = ce_per_pixel.mean()

        # Get weights (could be dynamic based on loss_strategy in future)
        dice_weight = self.dice_weight
        ce_weight = self.ce_weight

        # Combined total loss
        total_loss = (
            dice_weight * dice_loss_val +
            ce_weight * ce_loss_val
        )

        return {
            'total_loss': total_loss,
            'dice_loss': dice_loss_val,
            'ce_loss': ce_loss_val,
            'weights': {
                'dice': dice_weight,
                'ce': ce_weight,
            }
        }
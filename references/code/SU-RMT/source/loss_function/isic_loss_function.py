from typing import Dict, Any

import torch
from torch import nn
import torch.nn.functional as F

from .dice_loss import DiceLoss


class ISICLossFunction(nn.Module):
    """
    ISIC loss function.
    """
    def __init__(
        self,
        num_classes: int,
        dice_weight: float = 0.7,
        ce_weight: float = 0.3,
        loss_strategy: str = "Normal",
    ):
        """
        Initialize ISIC loss function.

        Args:
            num_classes: Number of classes.
            dice_weight: Dice weight.
            ce_weight: Cross entropy weight.
        """
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.current_epoch = 0

        self.dice_loss = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, Any]:
        """
        Calculate composite loss.
        """
        # one-hot construction
        if target.dim() == 3:
            n, h, w = target.shape
        elif target.dim() == 4 and target.size(1) == 1:
            # Compatible with (N,1,H,W)
            n, _, h, w = target.shape
            target = target.squeeze(1)
        else:
            raise ValueError(f"Unexpected target shape for one-hot: {target.shape}")

        target_one_hot = torch.zeros(
            (n, self.num_classes, h, w),
            device=target.device,
            dtype=torch.float32,
        )
        target_one_hot.scatter_(1, target.long().unsqueeze(1), 1.0)

        dice_loss_val = self.dice_loss(pred, target_one_hot)

        # Cross-Entropy: Avoid triggering nll_loss2d non-deterministic kernel
        # CE = - sum_c (y_onehot * log_softmax(pred)), then average over (N,H,W)
        log_probs = F.log_softmax(pred, dim=1)
        ce_per_pixel = -(target_one_hot * log_probs).sum(dim=1)
        ce_loss_val = ce_per_pixel.mean()

        ce_weight = self.ce_weight
        dice_weight = self.dice_weight

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

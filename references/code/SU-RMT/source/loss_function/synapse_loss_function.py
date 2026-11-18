from typing import Dict, Any, Tuple

import torch
from torch import nn

import torch.nn.functional as F
from .boundary_loss import BoundaryLoss
from .dice_loss import DiceLoss


class SynapseLossFunction(nn.Module):
    """
    Synapse loss function.
    """
    def __init__(
        self,
        num_classes: int,
        dice_weight: float = 0.7,
        ce_weight: float = 0.3,
        loss_strategy: str = "Normal",
    ):
        """
        Initialize Synapse loss function.

        Args:
            num_classes: Number of classes.
            dice_weight: Dice weight.
            ce_weight: Cross entropy weight.
            loss_strategy: Loss strategy
                - 'Normal': Normal loss.
                - 'StaticBoundaryLoss': Static boundary loss.
                - 'DynamicBoundaryLoss': Dynamic boundary loss.
                - 'DynamicLoss': Dynamic loss.
        """
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.current_epoch = 0
        self.loss_strategy = loss_strategy

        self.dice_loss = DiceLoss()
        self.boundary_loss = BoundaryLoss()

    def _compute_dynamic_weights(
        self,
        epoch: int,
        warmup_epochs: int = 40,
        transition_epochs: int = 20,
    ) -> Tuple[float, float]:
        """
        Compute dynamic weights for CE and Dice.

        Strategy:
        - Warmup stage [1, warmup_epochs]: fixed (0.7, 0.3)
        - Transition stage (warmup, warmup+transition]: linearly transition to (0.0, 1.0)
        - After transition: fixed (0.0, 1.0)
        """
        if epoch <= warmup_epochs:
            return 0.7, 0.3
        elif epoch <= warmup_epochs + transition_epochs:
            progress = (epoch - warmup_epochs) / transition_epochs
            ce_weight = 0.7 * (1.0 - progress)
            dice_weight = 0.3 + 0.7 * progress
            return ce_weight, dice_weight
        else:
            return 0.0, 1.0

    def _compute_dynamic_boundary_weights(
        self,
        epoch: int,
        warmup_epochs: int = 30,
        transition_epochs: int = 20,
    ) -> Tuple[float, float, float]:
        """
        Compute dynamic weights for CE, Dice and Boundary.

        Strategy:
        - Warmup stage: 0.7 CE + 0.3 Dice + 0.0 Boundary
        - Transition stage: linearly transition to 0.0 CE + 0.8 Dice + 0.2 Boundary
        - After transition: fixed 0.0 CE + 0.8 Dice + 0.2 Boundary
        """
        if epoch <= warmup_epochs:
            return 0.7, 0.3, 0.0
        elif epoch <= warmup_epochs + transition_epochs:
            progress = (epoch - warmup_epochs) / transition_epochs
            ce_weight = 0.7 * (1.0 - progress)
            dice_weight = 0.3 + 0.5 * progress
            boundary_weight = 0.2 * progress
            return ce_weight, dice_weight, boundary_weight
        else:
            return 0.0, 0.8, 0.2

    def set_epoch(self, epoch: int) -> None:
        """Set current training epoch."""
        self.current_epoch = epoch

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
        boundary_loss_val = torch.zeros((), device=pred.device, dtype=dice_loss_val.dtype)

        # Select weights and compute boundary loss based on strategy
        match self.loss_strategy:
            case "StaticBoundaryLoss":
                dice_weight = 0.7
                ce_weight = 0.3
                boundary_weight = 0.5
                # Only calculate boundary loss when needed
                pred_softmax = F.softmax(pred, dim=1)
                boundary_loss_val = self.boundary_loss(pred_softmax, target_one_hot)
            case "DynamicBoundaryLoss":
                ce_weight, dice_weight, boundary_weight = self._compute_dynamic_boundary_weights(
                    self.current_epoch,
                    warmup_epochs=30,
                    transition_epochs=20,
                )
                pred_softmax = F.softmax(pred, dim=1)
                boundary_loss_val = self.boundary_loss(pred_softmax, target_one_hot)
            case "DynamicLoss":
                ce_weight, dice_weight = self._compute_dynamic_weights(
                    self.current_epoch,
                    warmup_epochs=30,
                    transition_epochs=20,
                )
                boundary_weight = 0.0
            case "Normal":
                dice_weight = self.dice_weight
                ce_weight = self.ce_weight
                boundary_weight = 0.0

        # Combined total loss
        total_loss = (
            dice_weight * dice_loss_val +
            ce_weight * ce_loss_val +
            boundary_weight * boundary_loss_val
        )

        return {
            'total_loss': total_loss,
            'dice_loss': dice_loss_val,
            'ce_loss': ce_loss_val,
            'boundary_loss': boundary_loss_val,
            'weights': {
                'dice': dice_weight,
                'ce': ce_weight,
                'boundary': boundary_weight
            }
        }

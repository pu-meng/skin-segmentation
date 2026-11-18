import torch
from torch import nn
import torch.nn.functional as F


class BoundaryLoss(nn.Module):
    """
    Boundary loss function
    Reference MONAI's HausdorffDTLoss implementation, calculate boundary loss for each class separately
    """
    def __init__(self, alpha: float = 1.0, include_background: bool = False):
        """
        Args:
            alpha: Boundary weight strength
            include_background: Whether to include background class
        """
        super().__init__()
        self.alpha = alpha
        self.include_background = include_background

        # Sobel operator for boundary detection (single channel)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        self.register_buffer('sobel_x', sobel_x.unsqueeze(0).unsqueeze(0))
        self.register_buffer('sobel_y', sobel_y.unsqueeze(0).unsqueeze(0))

    def get_boundary_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Use Sobel operator to get single channel boundary map

        Args:
            x: Input tensor, shape (N, 1, H, W)

        Returns:
            Boundary strength map, same shape as input
        """
        # Ensure Sobel operator is on the correct device
        sobel_x = self.sobel_x.to(x.device)
        sobel_y = self.sobel_y.to(x.device)

        # Single channel convolution
        grad_x = F.conv2d(x, sobel_x, padding=1)  # pylint: disable=no-member, E1102
        grad_y = F.conv2d(x, sobel_y, padding=1)  # pylint: disable=no-member, E1102

        # Calculate gradient magnitude
        boundary = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

        return boundary

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate boundary loss
        Reference MONAI HausdorffDTLoss multi-class processing

        Args:
            pred: Predicted output, shape (N, C, H, W)
            target: Target label, shape (N, C, H, W)

        Returns:
            Boundary loss value
        """
        # If background is not included, skip the first channel
        if not self.include_background and pred.shape[1] > 1:
            pred = pred[:, 1:]
            target = target[:, 1:]

        # Calculate boundary loss for each class separately
        all_losses = []
        for i in range(pred.shape[1]):
            pred_ch = pred[:, [i]]  # Keep 4D (N, 1, H, W)
            target_ch = target[:, [i]]

            # Calculate boundary map
            pred_boundary = self.get_boundary_map(pred_ch)
            target_boundary = self.get_boundary_map(target_ch)

            # Calculate prediction error
            pred_error = (pred_ch - target_ch) ** 2

            # Boundary weight
            boundary_weight = torch.max(pred_boundary, target_boundary) * self.alpha

            # Weighted loss
            weighted_loss = pred_error * (1 + boundary_weight)

            # Calculate average loss for this class
            class_loss = weighted_loss.mean(dim=[2, 3], keepdim=True)
            all_losses.append(class_loss)

        # Merge all class losses
        total_loss = torch.cat(all_losses, dim=1)
        return total_loss.mean()

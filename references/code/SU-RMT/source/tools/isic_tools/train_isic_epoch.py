from typing import Optional

# 第三方库
import torch
from tqdm import tqdm
from torch.optim.swa_utils import AveragedModel
from torch.utils.tensorboard import SummaryWriter

# 自定义库
from source.loss_function import ISICLossFunction
from source.utils.log import Logger


def train_isic_epoch(
    now_epoch: int,
    train_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    num_classes: int = 2,
    ema_model: AveragedModel | None = None,
    writer: Optional[SummaryWriter] = None,
    logger: Optional[Logger] = None,
    loss_strategy: str = "Normal",
) -> None:
    """
    Train a single epoch of ISIC.

    Args:
        now_epoch (int): Current epoch.
        train_loader (torch.utils.data.DataLoader): Training data loader.
        model (torch.nn.Module): Current training model.
        optimizer (torch.optim.Optimizer): Current training optimizer.
        num_classes (int): Number of classes, default is 2.
        ema_model (AveragedModel | None): EMA model (optional).
        writer (SummaryWriter | None): TensorBoard writer (optional).
        logger (Logger | None): Structured logger (optional).

    Returns:
        float: Average total loss for the current epoch.
    """
    model.train()

    # Create composite loss: Dice + CE (same as Synapse)
    composite_loss = ISICLossFunction(
        num_classes=num_classes,
        dice_weight=0.7,
        ce_weight=0.3,
        loss_strategy=loss_strategy,
    )

    # Accumulator (aligned with Synapse)
    total_loss_sum = 0.0
    dice_loss_sum = 0.0
    ce_loss_sum = 0.0
    batch_count = 0

    # Progress bar (aligned with Synapse)
    pbar = tqdm(train_loader, desc=f"Epoch {now_epoch} Training")

    for _, (x, y) in enumerate(pbar):
        images = x.float().cuda()
        targets = y.cuda()

        # Compatible with (N,1,H,W) and (N,H,W) label formats
        if targets.dim() == 4 and targets.size(1) == 1:
            targets = targets.squeeze(1)
        targets = targets.long()

        # Forward
        outputs = model(images)

        loss_dict = composite_loss(outputs, targets)
        total_loss = loss_dict['total_loss']

        # Backward and update
        optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # EMA parameter update (optional)
        if ema_model is not None:
            ema_model.update_parameters(model)

        # Update progress bar information
        total_loss_sum += float(total_loss.item())
        dice_loss_sum += float(loss_dict['dice_loss'].item())
        ce_loss_sum += float(loss_dict['ce_loss'].item())
        batch_count += 1

        avg_loss = total_loss_sum / batch_count if batch_count > 0 else 0.0
        pbar.set_postfix({
            'Loss': f'{avg_loss:.4f}',
            'Dice': f'{dice_loss_sum/batch_count:.4f}',
            'CE': f'{ce_loss_sum/batch_count:.4f}'
        })

    # Summarize metrics
    metrics = {
        'total_loss': total_loss_sum / batch_count if batch_count > 0 else 0.0,
        'dice_loss': dice_loss_sum / batch_count if batch_count > 0 else 0.0,
        'ce_loss': ce_loss_sum / batch_count if batch_count > 0 else 0.0,
    }

    # Write to TensorBoard (aligned with Synapse)
    current_lr = optimizer.param_groups[0]['lr'] if optimizer and optimizer.param_groups else 0.0
    if writer is not None:
        writer.add_scalar("train/total_loss", metrics['total_loss'], now_epoch)
        writer.add_scalar("train/dice_loss", metrics['dice_loss'], now_epoch)
        writer.add_scalar("train/ce_loss", metrics['ce_loss'], now_epoch)
        writer.add_scalar("train/learning_rate", float(current_lr), now_epoch)

    # Structured logging
    if logger is not None:
        logger.log_metrics({
            'type': 'train',
            'epoch': now_epoch,
            'lr': float(current_lr),
            'total_loss': float(metrics['total_loss']),
            'dice_loss': float(metrics['dice_loss']),
            'ce_loss': float(metrics['ce_loss']),
        })

    return float(metrics['total_loss'])

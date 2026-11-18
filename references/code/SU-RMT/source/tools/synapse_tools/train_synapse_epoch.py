from typing import Dict, Optional

from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

from source.utils.log import Logger


def train_synapse_epoch(
    epoch: int,
    train_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    composite_loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ema_model: Optional[torch.nn.Module] = None,
    writer: Optional[SummaryWriter] = None,
    logger: Optional[Logger] = None,
) -> Dict[str, float]:
    """
    Synapse single epoch training (optional EMA).

    Args:
        epoch: Current epoch.
        train_loader: Training data loader.
        model: Training model.
        composite_loss: Composite loss function.
        optimizer: Optimizer.
        ema_model: Optional EMA model.

    Returns:
        Dictionary containing all loss values.
    """
    model.train()

    # Accumulator
    total_loss_sum = 0.0
    dice_loss_sum = 0.0
    ce_loss_sum = 0.0
    boundary_loss_sum = 0.0
    batch_count = 0
    has_boundary_loss = False

    # Progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training")

    for _, batch_data in enumerate(pbar):
        images = batch_data["image"].float().cuda()
        labels = batch_data["label"].long().cuda()

        # Zero gradients
        optimizer.zero_grad()

        # Forward
        outputs = model(images)
        loss_dict = composite_loss(outputs, labels)
        total_loss = loss_dict['total_loss']

        # Backward & Update
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # EMA update (optional)
        if ema_model is not None:
            ema_model.update_parameters(model)

        # Update progress bar
        total_loss_sum += total_loss.item()
        dice_loss_sum += loss_dict['dice_loss'].item()
        ce_loss_sum += loss_dict['ce_loss'].item()
        if 'boundary_loss' in loss_dict:
            boundary_loss_sum += loss_dict['boundary_loss'].item()
            has_boundary_loss = True
        batch_count += 1

        avg_loss = total_loss_sum / batch_count
        pbar.set_postfix({
            'Loss': f'{avg_loss:.4f}',
            'Dice': f'{dice_loss_sum/batch_count:.4f}',
            'CE': f'{ce_loss_sum/batch_count:.4f}'
        })

    metrics = {
        'total_loss': total_loss_sum / batch_count,
        'dice_loss': dice_loss_sum / batch_count,
        'ce_loss': ce_loss_sum / batch_count,
        'boundary_loss': boundary_loss_sum / batch_count,
    }

    # Write training metrics
    current_lr = optimizer.param_groups[0]['lr'] if optimizer and optimizer.param_groups else 0.0
    if writer is not None:
        writer.add_scalar("train/total_loss", metrics['total_loss'], epoch)
        writer.add_scalar("train/dice_loss", metrics['dice_loss'], epoch)
        writer.add_scalar("train/ce_loss", metrics['ce_loss'], epoch)
        if has_boundary_loss:
            writer.add_scalar("train/boundary_loss", metrics['boundary_loss'], epoch)
        writer.add_scalar("train/learning_rate", current_lr, epoch)

    if logger is not None:
        metrics_log = {
            'type': 'train',
            'epoch': epoch,
            'lr': current_lr,
            'total_loss': metrics['total_loss'],
            'dice_loss': metrics['dice_loss'],
            'ce_loss': metrics['ce_loss'],
        }
        if has_boundary_loss:
            metrics_log['boundary_loss'] = metrics['boundary_loss']
        logger.log_metrics(metrics_log)

    return metrics

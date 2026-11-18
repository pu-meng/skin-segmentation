import os

import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.tensorboard import SummaryWriter

from source.config import TrainConfig
from source.core import get_synapse_optimizer
from source.data.load_data.load_synapse import load_synapse_data
from source.loss_function import SynapseLossFunction
from source.tools.synapse_tools import (
    test_topk_models,
    validate_synapse,
    train_synapse_epoch
)
from source.utils.log import Logger


def train_synapse(
    args: TrainConfig,
    validation_results_csv: str,
    test_results_csv: str,
    model: torch.nn.Module,
    logger: Logger,
) -> None:
    """
    Synapse multi-organ segmentation training function
    Fast validation during training, based on Dice score selection of best model

    Args:
        args (TrainConfig): Command line arguments
        validation_results_csv (str): Validation results CSV file path
        test_results_csv (str): Test results CSV file path
        model (torch.nn.Module): Initialized model
        logger (Logger): Structured logger
    """
    # Initialize simplified loss function
    loss_function = SynapseLossFunction(
        num_classes=9,
        dice_weight=0.7,
        ce_weight=0.3,
        loss_strategy=args.loss_strategy,
    )

    # Load Synapse dataset
    train_loader, val_loader, test_loader = load_synapse_data(
        synapse_dataset_folder="./data/Synapse",
        batch_size=args.batch_size,
        num_workers=args.workers,
        list_dir="./data/lists_Synapse"
    )

    # Record dataset information to structured log
    logger.log_metrics({
        'type': 'dataset_info',
        'train_batches': len(train_loader),
        'val_batches': len(val_loader),
        'test_batches': len(test_loader)
    })

    optimizer, scheduler = get_synapse_optimizer(model, args)

    # Model save path (from command line arguments)
    base_model_path = os.path.join(
        args.log_root,
        f"{args.dataset_name}-{args.seed}",
        args.id
    )
    os.makedirs(base_model_path, exist_ok=True)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(base_model_path, "tensorboard"))

    # Dynamic management of Top-K best validation weights
    topk_checkpoints = []  # List: (epoch, checkpoint_path, dice_score), sorted by dice_score in descending order

    # Early stopping: stop if Top-K hasn't been updated for N epochs
    last_ckpt_update_epoch = 0  # the epoch index when Top-K last updated; 0 means never
    early_stop_epoch_patience = 30  # number of epochs since last update to trigger early stopping

    # Create EMA model (if enabled)
    ema_model = None
    if args.use_ema:
        device = next(model.parameters()).device
        ema_model = AveragedModel(
            model,
            device=device,
            multi_avg_fn=get_ema_multi_avg_fn(0.999),
            use_buffers=True,
        )

    # Training starts
    for epoch in range(1, args.epochs + 1):
        # Learning rate for this epoch (recorded internally by train_synapse_epoch)
        _ = optimizer.param_groups[0]['lr']

        # Update loss function's epoch information
        loss_function.set_epoch(epoch)

        _ = train_synapse_epoch(
            epoch=epoch,
            train_loader=train_loader,
            model=model,
            composite_loss=loss_function,
            optimizer=optimizer,
            ema_model=ema_model if args.use_ema else None,
            writer=writer,
            logger=logger,
        )

        # Update learning rate
        scheduler.step()

        # Validation strategy
        should_validate = False
        if (
            (epoch <= 100 and epoch % 20 == 0) or
            (110 < epoch <= 220 and epoch % 1 == 0) or
            (epoch > 220 and epoch % 2 == 0) or
            epoch == args.epochs
        ):
            should_validate = True

        if should_validate:
            # Save previous checkpoint set to detect updates
            prev_ckpt_paths = set([p for (_, p, _) in topk_checkpoints])

            # Validation wrapper call (contains logging, CSV writing, Top-K updating)
            validation_model = ema_model if args.use_ema else model
            _ = validate_synapse(
                val_loader=val_loader,
                eval_model=validation_model,
                orig_model=model,
                ema_model=ema_model,
                args=args,
                epoch=epoch,
                writer=writer,
                logger=logger,
                base_model_path=base_model_path,
                optimizer=optimizer,
                validation_results_csv=validation_results_csv,
                topk_checkpoints=topk_checkpoints,
                num_classes=9
            )

            # Early stopping check (epoch-based): update last update epoch if changed
            curr_ckpt_paths = set([p for (_, p, _) in topk_checkpoints])
            if curr_ckpt_paths != prev_ckpt_paths:
                last_ckpt_update_epoch = epoch

            # If it has been more than early_stop_epoch_patience epochs since last update, stop
            epochs_since_update = epoch - last_ckpt_update_epoch
            if epochs_since_update >= early_stop_epoch_patience:
                print(
                    f"Early stopping triggered: no Top-K update for {epochs_since_update} epochs."
                )
                if logger is not None:
                    logger.log_metrics({
                        'type': 'early_stopping',
                        'epoch': epoch,
                        'no_update_epochs': int(epochs_since_update),
                        'patience': int(early_stop_epoch_patience),
                        'topk_size': int(len(topk_checkpoints)),
                    })
                break

    # Test the top K best models
    test_topk_models(
        checkpoints=topk_checkpoints,
        test_loader=test_loader,
        model=model,
        results_csv=test_results_csv,
        logger=logger,
        num_classes=9,
        patch_size=args.out_size
    )

    writer.close()

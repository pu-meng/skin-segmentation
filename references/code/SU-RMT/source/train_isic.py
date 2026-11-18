import os

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from source.config import TrainConfig
from source.core import get_isic_optimizer
from source.data.load_data.load_isic import load_isic_data
from source.tools.isic_tools import (
    test_topk_models_isic,
    valid_isic_epoch,
    train_isic_epoch
)
from source.utils.log import Logger


def train_isic(
    args: TrainConfig,
    validation_results_csv: str,
    test_results_csv: str,
    model: torch.nn.Module,
    logger: Logger,
):
    """
    Training function for ISIC skin lesion segmentation (ISIC2017 & ISIC2018)

    Args:
        args(TrainConfig): Command line arguments
        validation_results_csv(str): Validation results CSV file path
        test_results_csv(str): Test results CSV file path
        model(torch.nn.Module): Initialized model
        logger(Logger): Structured logger

    Returns:
        None
    """
    # Top-K checkpoints (sorted by score from high to low)
    topk_checkpoints = []  # (epoch, checkpoint_path, score)

    match args.dataset_name:
        case 'ISIC2018':
            train_loader, valid_loader, test_loader, outer_ph2_test_loader = load_isic_data(
                isic_train_dataset_folder='./data/ISIC2018_npy_224_224',
                isic_valid_dataset_folder='./data/ISIC2018_validation_npy_224_224',
                isic_test_dataset_folder='./data/ISIC2018_test_npy_224_224',
                outer_ph2_dataset_folder='./data/PH2_npy_224_224',
                num_workers=args.workers,
                batch_size=args.batch_size,
                eval_batch_size=1,
                random_seed=args.seed
            )
        case 'ISIC2017':
            train_loader, valid_loader, test_loader, outer_ph2_test_loader = load_isic_data(
                isic_train_dataset_folder='./data/ISIC2017_npy_224_224',
                isic_valid_dataset_folder='./data/ISIC2017_validation_npy_224_224',
                isic_test_dataset_folder='./data/ISIC2017_test_npy_224_224',
                outer_ph2_dataset_folder='./data/PH2_npy_224_224',
                num_workers=args.workers,
                batch_size=args.batch_size,
                eval_batch_size=1,
                random_seed=args.seed
            )
        case _:
            raise ValueError(f"Unknown dataset: {args.dataset_name}")

    # Use unified optimizer and warmup scheduler
    optimizer, scheduler = get_isic_optimizer(model, args)

    base_model_path = os.path.join(
        args.log_root,
        f"{args.dataset_name}-{args.seed}",
        args.id
    )

    os.makedirs(base_model_path, exist_ok=True)
    writer = SummaryWriter(
        log_dir=os.path.join(
            args.log_root,
            f"{args.dataset_name}-{args.seed}",
            args.id,
            "tensorboard",
        )
    )

    logger.log_metrics({
        'type': 'dataset_info',
        'train_batches': len(train_loader),
        'val_batches': len(valid_loader),
        'test_batches': len(test_loader)
    })

    # Create EMA model (optional)
    ema_model = None
    if getattr(args, 'use_ema', False):
        device = next(model.parameters()).device
        ema_model = AveragedModel(
            model,
            device=device,
            multi_avg_fn=get_ema_multi_avg_fn(0.999),
            use_buffers=False,
        )

    # Early stopping counter: number of epochs without checkpoint update
    no_ckpt_update_epochs = 0
    early_stop_patience = 20  # Early stop if no checkpoint update for 20 epochs

    for epoch in range(1, args.epochs + 1):
        _ = train_isic_epoch(
            now_epoch=epoch,
            train_loader=train_loader,
            model=model,
            optimizer=optimizer,
            ema_model=ema_model,
            writer=writer,
            logger=logger,
            loss_strategy=args.loss_strategy,
        )

        scheduler.step()

        should_validate = False
        if (
            (epoch <= 5 and epoch % 5 == 0) or
            (5 < epoch <= 180 and epoch % 1 == 0) or
            epoch == args.epochs
        ):
            should_validate = True

        # Validation and checkpoint saving strategy (using the same process as valid_isic_epoch)
        if should_validate:
            # Save checkpoint set before validation to check if new weights are generated or replaced
            prev_ckpt_paths = set([p for (_, p, _) in topk_checkpoints])
            # Select model for validation (EMA or original model)
            validation_model = ema_model if getattr(args, 'use_ema', False) and ema_model is not None else model

            # Validation (internal log and Top-K maintenance)
            _ = valid_isic_epoch(
                now_epoch=epoch,
                valid_loader=valid_loader,
                eval_model=validation_model,
                validation_results_csv=validation_results_csv,
                writer=writer,
                logger=logger,
                base_model_path=base_model_path,
                optimizer=optimizer,
                topk_checkpoints=topk_checkpoints,
                orig_model=model,
                ema_model=ema_model,
                args=args
            )

            # Early stopping:
            # If the Top-K weight set does not change after validation, increment the counter, otherwise reset
            curr_ckpt_paths = set([p for (_, p, _) in topk_checkpoints])
            if curr_ckpt_paths == prev_ckpt_paths:
                no_ckpt_update_epochs += 1
            else:
                no_ckpt_update_epochs = 0

            # Early stopping: if the patience threshold is reached, stop training and switch to testing
            if no_ckpt_update_epochs >= early_stop_patience:
                print(f"Early stopping triggered: no checkpoint update for {no_ckpt_update_epochs} epochs.")
                if logger is not None:
                    logger.log_metrics({
                        'type': 'early_stopping',
                        'epoch': epoch,
                        'no_update_epochs': int(no_ckpt_update_epochs),
                        'patience': int(early_stop_patience),
                        'topk_size': int(len(topk_checkpoints)),
                    })
                break

    # Record final model selection (only Top-K)
    if len(topk_checkpoints) > 0:
        for i, (e, p, s) in enumerate(topk_checkpoints):
            logger.log_metrics({
                'type': 'final_model_selection',
                'rank': i + 1,
                'epoch': e,
                'composite_score': float(s),
                'checkpoint': os.path.basename(p)
            })

    # Test Top-K models directly and write to final CSV (same as Synapse, avoid redundancy)
    test_topk_models_isic(
        checkpoints=topk_checkpoints,
        test_loader=test_loader,
        ph2_loader=outer_ph2_test_loader,
        model=model,
        results_csv=test_results_csv,
    )
    writer.close()
    logger.close()

import os

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from source.config import TrainConfig
from source.core import get_polyp_optimizer
from source.data.load_data.load_polyp import load_polyp_data
from source.tools.polyp_tools import (
    test_topk_models_polyp,
    valid_polyp_epoch,
    train_polyp_epoch
)
from source.utils.log import Logger


def train_polyp(
    args: TrainConfig,
    validation_results_csv: str,
    test_results_csv: str,
    model: torch.nn.Module,
    logger: Logger,
):
    """
    Training function for Polyp segmentation.

    Args:
        args (TrainConfig): Command line arguments
        validation_results_csv (str): Validation results CSV file path
        test_results_csv (str): Test results CSV file path
        model (torch.nn.Module): Initialized model
        logger (Logger): Structured logger

    Returns:
        None
    """
    # Top-K checkpoints (sorted by score from high to low)
    topk_checkpoints = []  # (epoch, checkpoint_path, score)

    if args.dataset_name != 'Polyp':
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    train_loader, valid_loader, cvc300_loader, cvcclinicdb_loader, cvccolondb_loader, kvasir_loader, etislaribpolypdb_loader = load_polyp_data(  # noqa: E501
        train_dataset_folder='./data/PolypTrain_npy_224_224',
        valid_dataset_folder='./data/PolypValid_npy_224_224',
        cvc300_dataset_folder='./data/CVC300_npy_224_224',
        cvcclinicdb_dataset_folder='./data/CVCClinicDB_npy_224_224',
        cvccolondb_dataset_folder='./data/CVCColonDB_npy_224_224',
        kvasir_dataset_folder='./data/Kvasir_npy_224_224',
        etislaribpolypdb_dataset_folder='./data/ETISLaribPolypDB_npy_224_224',
        num_workers=args.workers,
        batch_size=args.batch_size,
        random_seed=args.seed,
    )

    # Use unified optimizer and warmup scheduler
    optimizer, scheduler = get_polyp_optimizer(model, args)

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
        'cvc300_batches': len(cvc300_loader),
        'cvcclinicdb_batches': len(cvcclinicdb_loader),
        'cvccolondb_batches': len(cvccolondb_loader),
        'kvasir_batches': len(kvasir_loader),
        'etislaribpolypdb_batches': len(etislaribpolypdb_loader),
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

    # Early stopping counter
    no_ckpt_update_epochs = 0
    early_stop_patience = 35

    for epoch in range(1, args.epochs + 1):
        _ = train_polyp_epoch(
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
            (epoch <= 80 and epoch % 10 == 0) or
            (80 < epoch <= 250 and epoch % 1 == 0) or
            epoch == args.epochs
        ):
            should_validate = True

        if should_validate:
            prev_ckpt_paths = set([p for (_, p, _) in topk_checkpoints])
            # Select model for validation (EMA or original model)
            validation_model = ema_model if getattr(args, 'use_ema', False) and ema_model is not None else model

            _ = valid_polyp_epoch(
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
                args=args,
            )

            curr_ckpt_paths = set([p for (_, p, _) in topk_checkpoints])
            if curr_ckpt_paths == prev_ckpt_paths:
                no_ckpt_update_epochs += 1
            else:
                no_ckpt_update_epochs = 0

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

    # Test Top-K models directly and write to final CSV
    loaders = {
        'CVC300': cvc300_loader,
        'CVCClinicDB': cvcclinicdb_loader,
        'CVCColonDB': cvccolondb_loader,
        'Kvasir': kvasir_loader,
        'ETISLaribPolypDB': etislaribpolypdb_loader,
    }
    test_topk_models_polyp(
        checkpoints=topk_checkpoints,
        loaders=loaders,
        model=model,
        results_csv=test_results_csv,
    )

    writer.close()
    logger.close()
    return None

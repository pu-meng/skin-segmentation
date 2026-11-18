from typing import Tuple

import torch
from lion_pytorch import Lion

from source.config import TrainConfig


def get_synapse_optimizer(
    model: torch.nn.Module, args: TrainConfig
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """
    Get optimizer and scheduler.

    Options:
      - warmup_cosine: warmup + single cosine (original behavior)
      - warmup_cosine_restarts: warmup + cosine with restarts (supports first restart at a given epoch)
    """
    optimizer = Lion(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=args.betas
    )

    # Scheduler
    warmup_epochs = 20

    def warmup_lambda(epoch: int) -> float:
        if epoch >= warmup_epochs:
            return 1.0
        return float(epoch + 1) / float(warmup_epochs)

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=warmup_lambda,
    )

    eta_min = args.lr * getattr(args, 'eta_min_factor', 0.05)

    if getattr(args, 'lr_scheduler', 'warmup_cosine') == 'warmup_cosine_restarts':
        # Cosine with restarts after warmup
        # Two ways to set the first restart:
        # 1) Absolute first restart epoch (restart_epoch > 0):
        #    After warmup, run cosine for (restart_epoch - warmup_epochs) epochs, then restart
        # 2) Otherwise use T_0/T_mult
        restart_epoch = int(getattr(args, 'restart_epoch', 0))
        if restart_epoch and restart_epoch > warmup_epochs:
            first_T = max(1, restart_epoch - warmup_epochs)
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=optimizer,
                T_0=first_T,
                T_mult=max(1, int(getattr(args, 'T_mult', 2))),
                eta_min=eta_min,
            )
        else:
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=optimizer,
                T_0=max(1, int(getattr(args, 'T_0', 50))),
                T_mult=max(1, int(getattr(args, 'T_mult', 2))),
                eta_min=eta_min,
            )
    else:
        # Original: single cosine after warmup
        cosine_epochs = max(1, args.epochs - warmup_epochs)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=max(1, cosine_epochs),
            eta_min=eta_min,
        )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer=optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    # Ensure first epoch uses LR=0
    for g in optimizer.param_groups:
        g['lr'] = 0.0

    return optimizer, scheduler


def get_isic_optimizer(
    model: torch.nn.Module, args: TrainConfig
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """
    Get optimizer and scheduler.

    Options:
      - warmup_cosine: warmup + single cosine (original behavior)
      - warmup_cosine_restarts: warmup + cosine with restarts (supports first restart at a given epoch)
    """
    optimizer = Lion(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=args.betas
    )

    # Scheduler
    warmup_epochs = 9 if args.dataset_name == "ISIC2018" else 1

    def warmup_lambda(epoch: int) -> float:
        if epoch >= warmup_epochs:
            return 1.0
        return float(epoch + 1) / float(warmup_epochs)

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=warmup_lambda,
    )

    eta_min = args.lr * getattr(args, 'eta_min_factor', 0.05)

    if getattr(args, 'lr_scheduler', 'warmup_cosine') == 'warmup_cosine_restarts':
        # Cosine with restarts after warmup
        # Two ways to set the first restart:
        # 1) Absolute first restart epoch (restart_epoch > 0):
        #    After warmup, run cosine for (restart_epoch - warmup_epochs) epochs, then restart
        # 2) Otherwise use T_0/T_mult
        restart_epoch = int(getattr(args, 'restart_epoch', 0))
        if restart_epoch and restart_epoch > warmup_epochs:
            first_T = max(1, restart_epoch - warmup_epochs)
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=optimizer,
                T_0=first_T,
                T_mult=max(1, int(getattr(args, 'T_mult', 2))),
                eta_min=eta_min,
            )
        else:
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=optimizer,
                T_0=max(1, int(getattr(args, 'T_0', 50))),
                T_mult=max(1, int(getattr(args, 'T_mult', 2))),
                eta_min=eta_min,
            )
    else:
        # Original: single cosine after warmup
        cosine_epochs = max(1, args.epochs - warmup_epochs)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=max(1, cosine_epochs),
            eta_min=eta_min,
        )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer=optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    # Ensure first epoch uses LR=0
    for g in optimizer.param_groups:
        g['lr'] = 0.0

    return optimizer, scheduler


def get_polyp_optimizer(
    model: torch.nn.Module, args: TrainConfig
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """
    Get optimizer and scheduler.

    Options:
      - warmup_cosine: warmup + single cosine (original behavior)
      - warmup_cosine_restarts: warmup + cosine with restarts (supports first restart at a given epoch)
    """
    optimizer = Lion(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=args.betas
    )

    # Scheduler
    warmup_epochs = 2

    def warmup_lambda(epoch: int) -> float:
        if epoch >= warmup_epochs:
            return 1.0
        return float(epoch + 1) / float(warmup_epochs)

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=warmup_lambda,
    )

    eta_min = args.lr * getattr(args, 'eta_min_factor', 0.05)

    if getattr(args, 'lr_scheduler', 'warmup_cosine') == 'warmup_cosine_restarts':
        # Cosine with restarts after warmup
        # Two ways to set the first restart:
        # 1) Absolute first restart epoch (restart_epoch > 0):
        #    After warmup, run cosine for (restart_epoch - warmup_epochs) epochs, then restart
        # 2) Otherwise use T_0/T_mult
        restart_epoch = int(getattr(args, 'restart_epoch', 0))
        if restart_epoch and restart_epoch > warmup_epochs:
            first_T = max(1, restart_epoch - warmup_epochs)
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=optimizer,
                T_0=first_T,
                T_mult=max(1, int(getattr(args, 'T_mult', 2))),
                eta_min=eta_min,
            )
        else:
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=optimizer,
                T_0=max(1, int(getattr(args, 'T_0', 50))),
                T_mult=max(1, int(getattr(args, 'T_mult', 2))),
                eta_min=eta_min,
            )
    else:
        # Original: single cosine after warmup
        cosine_epochs = max(1, args.epochs - warmup_epochs)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=max(1, cosine_epochs),
            eta_min=eta_min,
        )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer=optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    # Ensure first epoch uses LR=0
    for g in optimizer.param_groups:
        g['lr'] = 0.0

    return optimizer, scheduler

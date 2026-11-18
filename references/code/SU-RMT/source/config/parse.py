from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal, Tuple


ModelType = Literal["proposed"]
LossStrategy = Literal["Normal", "DynamicLoss", "DynamicBoundaryLoss", "StaticBoundaryLoss"]
DatasetName = Literal["ISIC2018", "ISIC2017", "Synapse", "Polyp"]


@dataclass
class TrainConfig:
    """
    Train configuration class
    Args:
        id (str): Model ID
        model_type (ModelType): Model type
        gpu_id (str): GPU ID
        out_size (Tuple[int, int]): Output size
        dataset_name (DatasetName): Dataset name
        epochs (int): Number of epochs
        loss_weights (Tuple[float, float, float]): Loss weights
        loss_strategy (LossStrategy): Loss strategy
        use_ema (bool): Whether to use EMA
        lr (float): Learning rate
        weight_decay (float): Weight decay
        betas (Tuple[float, float]): Betas
        batch_size (int): Batch size
        workers (int): Number of workers
        seed (int): Random seed
        top_k (int): Number of best checkpoints to keep during validation
    """
    id: str = "SURMT"
    model_type: ModelType = "proposed"
    gpu_id: str = "0"
    out_size: Tuple[int, int] = (224, 224)

    dataset_name: DatasetName = "ISIC2017"

    epochs: int = 300

    loss_weights: Tuple[float, float, float] = (0.70, 0.30, 0.00)
    loss_strategy: LossStrategy = "Normal"

    use_ema: bool = False
    lr: float = 3e-5
    weight_decay: float = 1.5e-3
    betas: Tuple[float, float] = (0.9, 0.99)
    batch_size: int = 24
    workers: int = 4

    seed: int = 1024
    top_k: int = 3
    debug_mode: bool = False
    log_root: str = "./log"

    lr_scheduler: str = "warmup_cosine"
    eta_min_factor: float = 0.1
    T_0: int = 50
    T_mult: int = 2
    restart_epoch: int = 0

    # -------------------------
    #  Build/parse command line
    # -------------------------
    @classmethod
    def build_arg_parser(cls: type[TrainConfig]) -> argparse.ArgumentParser:
        """
        Build argument parser for TrainConfig
        """
        parser = argparse.ArgumentParser(description="Skin lesion segmentation framework -- command line parameters")

        # 1) Basic parameters
        parser.add_argument("--id", default=cls.id, help="Model ID, optional models reference Models.__init__.py")
        parser.add_argument("--model_type", default=cls.model_type, choices=["proposed"], help="Model type")
        parser.add_argument("--gpu_id", default=cls.gpu_id, help="Specify GPU ID")
        parser.add_argument("--out_size", type=int, nargs=2, default=list(cls.out_size), help="Model input image size, e.g.: 224 224")

        # 2) Path/dataset
        parser.add_argument(
            "--dataset_name",
            default=cls.dataset_name,
            choices=["ISIC2018", "ISIC2017", "Synapse", "Polyp"],
            help="Dataset name",
        )

        # 3) Training configuration
        parser.add_argument("--epochs", type=int, default=cls.epochs, help="训练epoch总数，默认300")

        # 4) Loss related
        parser.add_argument(
            "--loss_weights", type=float, nargs=3, default=list(cls.loss_weights),
            help="Loss function weight coefficients [loss1, loss2, loss3], default [0.70, 0.30, 0.00]"
        )
        parser.add_argument(
            "--loss_strategy",
            type=str,
            default=cls.loss_strategy,
            choices=["Normal", "DynamicLoss", "DynamicBoundaryLoss", "StaticBoundaryLoss"],
            help="Loss function strategy: Normal / DynamicLoss / DynamicBoundaryLoss / StaticBoundaryLoss",
        )

        # 5) Enhanced training parameters
        parser.add_argument("--use_ema", action="store_true", help="Enable EMA weights and validation (only for Synapse)")
        parser.add_argument("--lr", type=float, default=cls.lr, help="Learning rate, default 3e-5")
        parser.add_argument("--weight_decay", type=float, default=cls.weight_decay, help="Weight decay, default 1e-3")
        parser.add_argument(
            "--betas", type=float, nargs=2, default=list(cls.betas),
            help="Optimizer parameters β1 β2, default 0.9 0.99"
        )
        parser.add_argument("--batch_size", type=int, default=cls.batch_size, help="Batch size, default 8")
        parser.add_argument("--workers", type=int, default=cls.workers, help="Data loader worker count, default 4")

        # 6) 其他
        parser.add_argument("--seed", type=int, default=cls.seed, help="Random seed, default 1024")
        parser.add_argument("--top_k", type=int, default=cls.top_k, help="Number of best models to keep during validation, default 3")
        parser.add_argument("--log_root", type=str, default=cls.log_root, help="Training logs and weights root directory, default ./log")
        parser.add_argument(
            "--debug_mode",
            action="store_true",
            help="Debug mode: test and fill CSV immediately after validation (large overhead)",
        )

        # 7) Learning rate scheduler enhancement
        parser.add_argument(
            "--lr_scheduler",
            type=str,
            default=cls.lr_scheduler,
            choices=["warmup_cosine", "warmup_cosine_restarts"],
            help="Learning rate scheduler strategy: warmup_cosine (default) or warmup_cosine_restarts (cosine restarts)",
        )
        parser.add_argument(
            "--eta_min_factor",
            type=float,
            default=cls.eta_min_factor,
            help="Minimum learning rate factor: eta_min = lr * eta_min_factor, default 0.1",
        )
        parser.add_argument(
            "--T_0", type=int, default=cls.T_0,
            help="Initial period length of CosineAnnealingWarmRestarts (unit: epoch), default 50"
        )
        parser.add_argument(
            "--T_mult", type=int, default=cls.T_mult,
            help="Period multiplier of CosineAnnealingWarmRestarts, default 2"
        )
        parser.add_argument(
            "--restart_epoch", type=int, default=cls.restart_epoch,
            help="Absolute epoch of first restart (>0 enabled, prior to T_0), e.g. 220 means restart at the 220th epoch"
        )

        return parser

    @classmethod
    def parse_args(cls) -> "TrainConfig":
        """Parse command line and return strongly typed configuration object."""
        parser = cls.build_arg_parser()
        ns = parser.parse_args()
        return cls(
            id=ns.id,
            model_type=ns.model_type,
            gpu_id=ns.gpu_id,
            out_size=(ns.out_size[0], ns.out_size[1]) if isinstance(ns.out_size, (list, tuple)) else tuple(ns.out_size),
            dataset_name=ns.dataset_name,
            epochs=ns.epochs,
            loss_weights=(ns.loss_weights[0], ns.loss_weights[1], ns.loss_weights[2]),
            loss_strategy=ns.loss_strategy,
            use_ema=bool(ns.use_ema),
            lr=ns.lr,
            weight_decay=ns.weight_decay,
            betas=(ns.betas[0], ns.betas[1]),
            batch_size=ns.batch_size,
            workers=ns.workers,
            seed=ns.seed,
            top_k=max(1, int(ns.top_k)),
            log_root=ns.log_root,
            lr_scheduler=ns.lr_scheduler,
            eta_min_factor=ns.eta_min_factor,
            T_0=ns.T_0,
            T_mult=ns.T_mult,
            restart_epoch=ns.restart_epoch,
            debug_mode=bool(ns.debug_mode),
        )


def get_args() -> TrainConfig:
    """
    Get arguments from command line
    """
    return TrainConfig.parse_args()

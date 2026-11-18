# src/lightning/seg_module.py
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from monai.losses import DiceCELoss
from omegaconf import DictConfig
from torchmetrics import MetricCollection
import pytorch_lightning as pl
import torch.nn.functional as F

from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex, BinaryRecall, BinaryPrecision, \
    MulticlassF1Score, MulticlassJaccardIndex
from ..metrics.distance_metric import BinaryASD, BinaryHD95


class SegLitModule(pl.LightningModule):
    """Unified LightningModule for 2D medical image segmentation (binary & multiclass)."""

    def __init__(self, model: nn.Module, cfg: DictConfig) -> None:
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.save_hyperparameters(ignore=["model"])

        # ---- task ----
        self.task = str(cfg.data.task.lower())  # "binary" | "multiclass"
        if self.task not in ("binary", "multiclass"):
            raise ValueError(f"Unknown cfg.data.task='{self.task}', must be 'binary' or 'multiclass'.")

        self.thr = cfg.metrics.threshold

        # ---- loss ----
        if self.task == "binary":
            self.criterion = DiceCELoss(
                sigmoid=True,
                lambda_dice=float(self.cfg.loss.lambda_dice),
                lambda_ce=float(self.cfg.loss.lambda_ce),
            )
        else:
            self.num_classes = int(cfg.data.num_classes)  # including background
            self.include_background = bool(cfg.data.get("include_background", False))
            self.ignore_index = None if self.include_background else 0

            self.criterion = DiceCELoss(
                softmax=True,
                to_onehot_y=False,
                include_background=True,
                lambda_dice = float(self.cfg.loss.lambda_dice),
                lambda_ce = float(self.cfg.loss.lambda_ce),
            )

        # ---- metrics (separate for each stage) ----
        self.train_metrics = self.build_metrics()
        self.val_metrics = self.build_metrics()
        self.test_metrics = self.build_metrics()

    def build_metrics(self) -> MetricCollection:
        if self.task == "binary":
            return MetricCollection({
                "Dice": BinaryF1Score(threshold=self.thr),
                "IoU": BinaryJaccardIndex(threshold=self.thr),
                "HD95": BinaryHD95(threshold=self.thr),
                "ASD": BinaryASD(threshold=self.thr),
                "Recall": BinaryRecall(threshold=self.thr),
                "Precision": BinaryPrecision(threshold=self.thr),
            })
        else:
            return MetricCollection({
                "mDice": MulticlassF1Score(
                    num_classes=self.num_classes, average="macro", ignore_index=self.ignore_index
                ),
                "mIoU": MulticlassJaccardIndex(
                    num_classes=self.num_classes, average="macro", ignore_index=self.ignore_index
                ),
            })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # ---------- helpers ----------
    def _get_metrics(self, stage: str) -> MetricCollection:
        if stage == "train":
            return self.train_metrics
        if stage == "val":
            return self.val_metrics
        if stage == "test":
            return self.test_metrics
        raise ValueError(stage)

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor | Dict[str, torch.Tensor]:
        x = batch["image"]
        y = batch["mask"]

        logits = self(x)
        final_logits = logits[0] if isinstance(logits, list) else logits

        if self.task == "multiclass":
            y_oh = F.one_hot(y.long(), num_classes=self.num_classes).permute(0, 3, 1, 2).float()
            assert final_logits.shape[1] == self.num_classes, \
                f"multiclass requires logits channels C={self.num_classes}, but got {final_logits.shape}"
            target_for_loss = y_oh
        else:
            target_for_loss = y

        if isinstance(logits, list):
            weights = [1.0, 0.5, 0.25, 0.125]

            loss = 0.0
            for i, (pred, w) in enumerate(zip(logits, weights)):
                loss += w * self.criterion(pred, target_for_loss)
        else:
            loss = self.criterion(logits, target_for_loss)

        metrics = self._get_metrics(stage)

        if self.task == "binary":
            pred = torch.sigmoid(final_logits).detach()
            tgt = (y >= self.thr).int()  # (B,1,H,W) int
            metrics.update(pred, tgt)
        else:
            pred = torch.argmax(final_logits, dim=1).detach()  # (B,H,W) long
            tgt = y.long()
            metrics.update(pred, tgt)

        self.log(
            f"{stage}/loss",
            loss.detach(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=x.shape[0]
        )
        if stage == "test":
            return {
                "loss": loss.detach(),
                "preds": pred.detach(),
                "mask": tgt.detach(),
                "image": x.detach(),
                "filenames": batch["id"]
            }
        return loss

    # ---------- lightning hooks ----------
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, stage="val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="test")

    def on_train_epoch_end(self) -> None:
        self._log_epoch_metrics("train")

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics("val")

    def on_test_epoch_end(self) -> None:
        self._log_epoch_metrics("test")

    def _log_epoch_metrics(self, stage: str) -> None:
        metrics = self._get_metrics(stage)
        metric_results = metrics.compute()

        for metric_name, value in metric_results.items():
            self.log(
                f"{stage}/{metric_name}",
                value,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True
            )

        metrics.reset()

    def configure_optimizers(self):
        opt_cfg = self.cfg.optim
        lr0 = float(opt_cfg.lr)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr0, weight_decay=opt_cfg.weight_decay)

        sched_name = opt_cfg.scheduler.lower()  # "poly" | "cosine" | "plateau"
        min_lr = float(opt_cfg.min_lr)

        if sched_name == "poly":
            def poly(step: int):
                total_steps = int(self.trainer.estimated_stepping_batches)
                if total_steps <= 1:
                    return 1.0
                step = min(step, total_steps)
                factor = (1.0 - step / total_steps) ** 0.9
                return max(min_lr / lr0, factor)

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly)

        elif sched_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=min_lr,
            )

        elif sched_name == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.cfg.trainer.early_stopping.mode,
                factor=0.5,
                patience=10,
                threshold=1e-4,
                min_lr=min_lr,
            )

        else:
            raise ValueError(f"Unknown scheduler '{sched_name}'. Choose from: poly | cosine | plateau")

        print(f"[Optimization scheduler] {sched_name} lr0={lr0:.4f}, min_lr={min_lr:.6f}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.cfg.trainer.early_stopping.monitor if sched_name == "plateau" else None,
                "interval": "epoch" if sched_name in ("plateau", "cosine") else "step",
                "frequency": 1,
                "name": f"lr-{sched_name}"
            },
        }

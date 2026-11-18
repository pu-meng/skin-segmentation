# src/callbacks/earlystop_print.py
from __future__ import annotations

import pytorch_lightning as pl

class EarlyStopPrinter(pl.Callback):

    def __init__(self, monitor: str, mode: str = "min", patience: int = 20):
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.best = None
        self.wait_count = 0

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        metrics = trainer.callback_metrics
        if self.monitor not in metrics:
            return
        val = metrics[self.monitor]
        try:
            val = float(val.detach().cpu().item() if hasattr(val, "detach") else val)
        except Exception:
            return

        improved = False
        if self.best is None:
            improved = True
        else:
            improved = (val < self.best) if self.mode == "min" else (val > self.best)

        if improved:
            self.best = val
            self.wait_count = 0
            trainer.print(f"[EarlyStop] {self.monitor} improved -> best={self.best:.4f}")
        else:
            self.wait_count += 1

        if self.wait_count >= self.patience:
            trainer.print(f"[EarlyStop] Early stopping triggered! No improvement for {self.patience} epochs.")
            trainer.should_stop = True

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """训练结束时输出最终结果"""
        if self.best is not None:
            trainer.print(f"[EarlyStop] Training finished. Best {self.monitor}: {self.best:.4f}")
        else:
            trainer.print("[EarlyStop] Training finished. No validation metrics were recorded.")

# src/train.py
from __future__ import annotations

import argparse

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

from src.callbacks.earlystop_print import EarlyStopPrinter
from src.data.datamodule import MultiDatasetDataModule
from src.models.registry import build_model, get_active_model
from src.lightning.seg_module import SegLitModule
from src.callbacks.log_visualize import LoggerVisCallback
from src.callbacks.rich_progress import CustomRichProgressBar


def main():
    cfg = OmegaConf.load(args.config)
    pl.seed_everything(int(cfg.seed), workers=True)

    # data
    dm = MultiDatasetDataModule(cfg)

    # model
    active_model_cfg = get_active_model(cfg)
    model = build_model(active_model_cfg["name"], **active_model_cfg.get("kwargs", {}))
    lit = SegLitModule(model=model, cfg=cfg)

    # logging
    tb_logger = TensorBoardLogger(save_dir=cfg.logging.save_dir, name=cfg.experiment_name, default_hp_metric=False,
                                  version=cfg.data.activate_dataset
                                  )
    csv_logger = CSVLogger(save_dir=cfg.logging.save_dir, name=cfg.experiment_name, version=tb_logger.version)
    print(f"[Logger] Tensorboard is supported, try to use by running `tensorboard --logdir={cfg.logging.save_dir}`")

    # callbacks
    ckpt = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        filename=f"e{{epoch:03d}}-dice{{val/Dice:.4f}}-hd{{val/HD95:.4f}}-ds{cfg.data.activate_dataset}",
        auto_insert_metric_name=False,
        save_last=True
    )
    lr_mon = LearningRateMonitor(logging_interval="step")

    vis_cb = LoggerVisCallback(
        fixed_ids=None,
        fixed_first_n=4,
        topk=cfg.logging.topk,
        threshold=cfg.metrics.threshold,
    )

    progress = CustomRichProgressBar(bar_width=15)

    early_stop = EarlyStopping(
        monitor=cfg.trainer.early_stopping.monitor,
        mode=cfg.trainer.early_stopping.mode,
        patience=cfg.trainer.early_stopping.patience,
        min_delta=cfg.trainer.early_stopping.min_delta,
        verbose=False,
    )

    trainer = pl.Trainer(
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        logger=[tb_logger, csv_logger],
        callbacks=[
            ckpt,
            early_stop,
            EarlyStopPrinter(
                monitor=cfg.trainer.early_stopping.monitor,
                mode=cfg.trainer.early_stopping.mode,
                patience=cfg.trainer.early_stopping.patience
            ),
            lr_mon,
            progress,
            vis_cb,
        ],
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        deterministic=cfg.trainer.deterministic,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        fast_dev_run=False
    )

    trainer.fit(lit, datamodule=dm)

    if ckpt.best_model_path:
        print("\n[Tester] Running evaluation on the best model checkpoint (selected by validation loss)...")
        trainer.test(lit, datamodule=dm, ckpt_path="best")

        print("\n[Tester] Running evaluation on the last model checkpoint (end of training)...")
        trainer.test(lit, datamodule=dm, ckpt_path="last")
    else:
        trainer.test(lit, datamodule=dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    main()

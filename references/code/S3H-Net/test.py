from __future__ import annotations

import argparse

import pytorch_lightning as pl
from omegaconf import OmegaConf

from src.callbacks.test_callbacks import TestVisCallback
from src.data.datamodule import MultiDatasetDataModule
from src.models.registry import build_model, get_active_model
from src.lightning.seg_module import SegLitModule
from src.callbacks.rich_progress import CustomRichProgressBar


def test():
    cfg = OmegaConf.load(args.config)
    pl.seed_everything(int(cfg.seed), workers=True)

    dm = MultiDatasetDataModule(cfg)

    active_model_cfg = get_active_model(cfg)
    model_arch = build_model(active_model_cfg["name"], **active_model_cfg.get("kwargs", {}))

    print(f"[*] Loading checkpoint from: {args.ckpt_path}")
    lit = SegLitModule.load_from_checkpoint(
        checkpoint_path=args.ckpt_path,
        model=model_arch,
        cfg=cfg
    )

    progress = CustomRichProgressBar(bar_width=15)

    vis = TestVisCallback(save_dir="outputs", enabled=True, num_classes=cfg.data.num_classes,
                          dataset_name=cfg.data.activate_dataset, model_name=cfg.model.active_model)

    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=args.devices,
        precision=cfg.trainer.precision,
        callbacks=[progress, vis],
        deterministic=True
    )

    print(f"\n[Evaluate] Starting test for model: {active_model_cfg['name']}")
    trainer.test(lit, datamodule=dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to the config file (same as training)")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the .ckpt file to evaluate")
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs to use")
    args = parser.parse_args()
    test()
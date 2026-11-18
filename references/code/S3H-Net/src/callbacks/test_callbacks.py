# src/callbacks/test_callbacks.py
from __future__ import annotations

import csv
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image


class TestVisCallback(pl.Callback):
    def __init__(self,
                 save_dir: str = "outputs",
                 dataset_name: str = "",
                 model_name: str = "",
                 num_samples: int = float(np.inf),
                 enabled: bool = False,
                 overlay_on_image: bool = False,
                 num_classes: int = 1,
                 class_colors: Optional[Dict[int, List[int]]] = None,
                 alpha: float = 0.4):
        super().__init__()
        self.enabled = enabled

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{timestamp}-{dataset_name}-{model_name}"
        self.save_dir = Path(save_dir) / dir_name
        self.img_dir = self.save_dir / "plots"
        self.csv_path = self.save_dir / "per_sample_metrics.csv"

        self.max_samples = num_samples
        self.samples_count = 0
        self.test_metrics = None
        self.alpha = alpha
        self.overlay_on_image = overlay_on_image
        self.num_classes = num_classes

        self.color_map = self._get_default_colors()
        if class_colors is not None:
            self.color_map.update(class_colors)

    @staticmethod
    def _get_default_colors() -> Dict[int, List[int]]:
        """定义默认的类别颜色 (RGB)"""
        return {
            1: [255, 0, 0],  # Red
            2: [0, 255, 0],  # Green
            3: [0, 0, 255],  # Blue
            4: [255, 255, 0],  # Yellow
            5: [255, 0, 255],  # Magenta
            6: [0, 255, 255],  # Cyan
            7: [128, 0, 0],  # Maroon
            8: [0, 128, 0],  # Dark Green
            9: [0, 0, 128],  # Navy
            10: [128, 128, 0]  # Olive
        }

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.enabled:
            return

        self.img_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(pl_module, 'build_metrics'):
            self.test_metrics = pl_module.build_metrics().to(pl_module.device)
            with open(self.csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                metric_names = list(self.test_metrics.keys())
                header = ["sample_idx"] + metric_names
                writer.writerow(header)

    def on_test_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule,
                          outputs: Dict[str, torch.Tensor], batch: Any,
                          batch_idx: int, dataloader_idx: int = 0) -> None:
        if not self.enabled:
            return

        preds = outputs["preds"]
        masks = outputs["mask"]
        imgs = outputs["image"]
        filenames = outputs["filenames"]

        batch_size = preds.shape[0]

        for i in range(batch_size):
            try:
                p = preds[i:i + 1].detach()
                m = masks[i:i + 1].detach()
                img_tensor = imgs[i].detach()
                raw_name = str(filenames[i])
                fname = Path(raw_name).stem

                if self.test_metrics:
                    self.test_metrics.reset()
                    self.test_metrics.update(p.to(pl_module.device), m.to(pl_module.device))
                    results = self.test_metrics.compute()

                    row = {"sample_idx": fname}
                    for k, v in results.items():
                        row[k] = f"{v.item():.4f}"
                    self._write_to_csv(row)

                if self.samples_count < self.max_samples:
                    self._save_overlay(img_tensor, m, p, fname)
                    self.samples_count += 1

            except Exception as e:
                print(f"Error saving sample at batch {batch_idx}, index {i}: {e}")
                continue

    def _write_to_csv(self, row_dict: dict) -> None:
        file_exists = self.csv_path.exists()
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row_dict.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_dict)

    def _save_overlay(self, img: torch.Tensor, mask: torch.Tensor, pred: torch.Tensor, filename: str) -> None:
        """
        核心绘图逻辑
        img: (C, H, W)
        mask: (1, H, W) GT
        pred: (C, H, W) Logits/Probs
        """
        img_np = img.cpu().numpy()
        img_np = np.transpose(img_np, (1, 2, 0))

        if img_np.max() <= 1.0 + 1e-5 and img_np.min() >= -2.0:
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)

        if img_np.shape[2] == 1:
            img_np = np.repeat(img_np, 3, axis=2)


        final_img = img_np.copy()
        overlay = np.zeros_like(img_np)

        if self.num_classes <= 2:
            # GT
            m = mask.cpu().squeeze().numpy().astype(bool)

            # Pred (logits or probs)
            p_tensor = pred.cpu()
            if p_tensor.shape[1] > 1:  # (1, C, H, W) -> (C, H, W) -> argmax
                p = torch.argmax(p_tensor.squeeze(0), dim=0).numpy().astype(bool)
            else:  # Sigmoid
                p = (p_tensor.squeeze() > 0.5).numpy().astype(bool)

            # TP/FN/FP Logic
            tp_mask = m & p
            fn_mask = m & (~p)
            fp_mask = (~m) & p

            overlay[tp_mask] = [0, 255, 0]  # Green TP
            overlay[fn_mask] = [255, 0, 0]  # Red FN
            overlay[fp_mask] = [0, 0, 255]  # Blue FP

            if self.overlay_on_image:
                roi = tp_mask | fn_mask | fp_mask
                if roi.any():
                    final_img[roi] = (
                            img_np[roi] * (1 - self.alpha) +
                            overlay[roi] * self.alpha
                    ).astype(np.uint8)
            else:
                final_img = overlay

        else:
            p_tensor = pred.cpu().squeeze(0)  # (C, H, W)

            if p_tensor.ndim == 3 and p_tensor.shape[0] > 1:
                pred_map = torch.argmax(p_tensor, dim=0).numpy()
            else:
                pred_map = p_tensor.numpy().astype(int)

            unique_classes = np.unique(pred_map)
            has_foreground = False

            for cls_idx in unique_classes:
                if cls_idx == 0: continue

                has_foreground = True
                class_mask = (pred_map == cls_idx)

                color = self.color_map.get(cls_idx, [255, 255, 255])

                overlay[class_mask] = color

            if self.overlay_on_image and has_foreground:
                fg_mask = (pred_map > 0)

                final_img[fg_mask] = (
                        img_np[fg_mask] * (1 - self.alpha) +
                        overlay[fg_mask] * self.alpha
                ).astype(np.uint8)
            elif not self.overlay_on_image:
                final_img = overlay

        save_path = self.img_dir / f"{filename}.png"
        Image.fromarray(final_img).save(save_path)
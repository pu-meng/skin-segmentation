#!/usr/bin/env python3
"""Project-split FocalTransNet ISIC reproduction entry point.

The frozen author implementation supplies the model and augmentation family.
Training selects checkpoints on validation Global Dice; test is only opened by
the explicit --mode test command after training is complete.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "references/code/FocalTransNet"
CACHE = ROOT / "data/processed/isic2018_256_dedup_seed42_20261031"
SPLIT = ROOT / "data/processed/isic2018_dedup_seed42_20261031/split.csv"
CONFIG_PATH = Path(__file__).with_name("config.json")
sys.path.insert(0, str(REF))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class IsicSplit(Dataset):
    def __init__(self, split: str, training: bool):
        if split not in {"train", "val", "test"}:
            raise ValueError(split)
        with (CACHE / "split.csv").open(newline="") as f:
            self.rows = [r for r in csv.DictReader(f) if r["split"] == split]
        # The cache adds cache_index/dx columns, so compare its recorded source
        # digest and the ordered image_id/split mapping rather than whole CSV bytes.
        manifest = json.loads((CACHE / "cache_complete.json").read_text())
        if sha256(SPLIT) != manifest["dedup_split_sha256"]:
            raise RuntimeError("deduplicated split digest differs from the verified image cache")
        with SPLIT.open(newline="") as f:
            source_rows = [r for r in csv.DictReader(f) if r["split"] == split]
        if [(r["image_id"], r["split"]) for r in self.rows] != [
                (r["image_id"], r["split"]) for r in source_rows]:
            raise RuntimeError("cached split IDs/order differ from frozen deduplicated split")
        self.images = np.load(CACHE / "images.npy", mmap_mode="r")
        self.masks = np.load(CACHE / "masks.npy", mmap_mode="r")
        self.training = training
        self.augmenter = None
        if training:
            try:
                # imgaug 0.4.0 predates NumPy 2. Define the removed dtype
                # catalog locally so the frozen official augmentation code
                # remains usable in the project's NumPy 2 environment.
                if not hasattr(np, "sctypes"):
                    np.sctypes = {
                        "int": [np.int8, np.int16, np.int32, np.int64],
                        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
                        "float": [np.float16, np.float32, np.float64],
                        "complex": [np.complex64, np.complex128],
                        "others": [bool, object, bytes, str, np.void],
                    }
                import imgaug.augmenters as iaa
            except ImportError as exc:
                raise RuntimeError("Install official augmentation dependency imgaug==0.4.0 before training") from exc
            self.augmenter = iaa.SomeOf((0, 4), [
                iaa.Flipud(0.5), iaa.Fliplr(0.5),
                iaa.AdditiveGaussianNoise(scale=0.005 * 255),
                iaa.GaussianBlur(sigma=(1.0)),
                iaa.LinearContrast((0.5, 1.5), per_channel=0.5),
                iaa.Affine(scale={"x": (0.5, 2), "y": (0.5, 2)}),
                iaa.Affine(rotate=(-40, 40)), iaa.Affine(shear=(-16, 16)),
                iaa.PiecewiseAffine(scale=(0.008, 0.03)),
                iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
            ], random_order=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        i = int(row["cache_index"])
        image = np.asarray(self.images[i]).copy()
        mask = (np.asarray(self.masks[i]) > 0).astype(np.uint8)
        if self.augmenter is not None:
            import imgaug as ia
            det = self.augmenter.to_deterministic()
            image = det.augment_image(image)
            seg = ia.SegmentationMapsOnImage(mask[..., None], shape=image.shape)
            mask = det.augment_segmentation_maps(seg).get_arr()[:, :, 0].astype(np.uint8)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
        image = torch.from_numpy(image.transpose(2, 0, 1).copy()).float().div_(255).sub_(0.5).div_(0.5)
        mask = torch.from_numpy((mask > 0).astype(np.int64))
        return image, mask, row["image_id"]


class OfficialIsicLoss(torch.nn.Module):
    """Equivalent to the frozen author's ISICLoss and DiceLoss implementations."""
    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target.long())
        probs = torch.softmax(logits, dim=1)
        onehot = F.one_hot(target.long(), num_classes=2).permute(0, 3, 1, 2).to(probs.dtype)
        dice_terms = []
        for cls in range(2):
            p, y = probs[:, cls], onehot[:, cls]
            dice_terms.append(1 - (2 * (p * y).sum() + 1e-5) /
                              ((p * p).sum() + (y * y).sum() + 1e-5))
        dice = torch.stack(dice_terms).mean()
        return 0.8 * ce + 0.2 * dice, ce.detach(), dice.detach()


def build_model():
    from networks.Net import Net
    # The frozen author model defaults to allocating internal tensors on cuda:0.
    # Construct on CPU so an expctl-selected GPU is honored, then move normally.
    return Net(img_size=224, dim_in=3, dim_out=2, scale_factor=2, embed_dim=96,
               device="cpu", split_size=[1, 2, 2, 7],
               num_heads=[2, 4, 8, 16], depth=[1, 2, 4, 2])


def global_dice(model, loader, device, use_mirror4=False):
    model.eval()
    tp = fp = fn = 0
    with torch.inference_mode():
        for images, masks, _ in loader:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            logits = model(images)
            if use_mirror4:
                logits = (logits + model(images.flip(-1)).flip(-1) +
                          model(images.flip(-2)).flip(-2) +
                          model(images.flip((-2, -1))).flip((-2, -1))) / 4
            pred = logits.argmax(1).bool()
            truth = masks.bool()
            tp += int((pred & truth).sum())
            fp += int((pred & ~truth).sum())
            fn += int((~pred & truth).sum())
    return {"global_dice": 100.0 * (2 * tp) / max(2 * tp + fp + fn, 1),
            "tp": tp, "fp": fp, "fn": fn}


def make_loader(split, training, cfg):
    ds = IsicSplit(split, training)
    return DataLoader(ds, batch_size=cfg["training"]["batch_size"] if training else 1,
                      shuffle=training, num_workers=cfg["training"]["num_workers"] if training else 2,
                      pin_memory=True, drop_last=False, persistent_workers=False)


def train(run_dir: Path, device: torch.device):
    cfg = json.loads(CONFIG_PATH.read_text())
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg["author_seed"])
    train_loader = make_loader("train", True, cfg)
    val_loader = make_loader("val", False, cfg)
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5, betas=(0.9, 0.999), weight_decay=0.03)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20000)
    criterion = OfficialIsicLoss()
    history_path = run_dir / "history.csv"
    best = -1.0
    with history_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_global_dice", "lr", "seconds"])
        writer.writeheader()
        for epoch in range(1, 101):
            started = time.time()
            model.train()
            losses = []
            for images, masks, _ in train_loader:
                images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss, _, _ = criterion(logits, masks)
                loss.backward()
                optimizer.step()
                scheduler.step()
                losses.append(float(loss.detach()))
            val = global_dice(model, val_loader, device)
            row = {"epoch": epoch, "train_loss": float(np.mean(losses)),
                   "val_global_dice": val["global_dice"], "lr": optimizer.param_groups[0]["lr"],
                   "seconds": time.time() - started}
            writer.writerow(row); f.flush()
            if val["global_dice"] > best:
                best = val["global_dice"]
                tmp = run_dir / "best.pt.tmp"
                torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                            "val_global_dice": best, "config": cfg}, tmp)
                os.replace(tmp, run_dir / "best.pt")
            print(json.dumps(row, ensure_ascii=False), flush=True)
    (run_dir / "train_complete.json").write_text(json.dumps({"epochs": 100, "best_val_global_dice": best}, indent=2) + "\n")


def test(run_dir: Path, device: torch.device):
    cfg = json.loads(CONFIG_PATH.read_text())
    checkpoint = run_dir / "best.pt"
    if not checkpoint.is_file() or not (run_dir / "train_complete.json").is_file():
        raise RuntimeError("Test is locked until 100-epoch training completes and best.pt exists")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model().to(device)
    model.load_state_dict(state["state_dict"])
    loader = make_loader("test", False, cfg)
    outputs = {"none": global_dice(model, loader, device),
               "mirror4": global_dice(model, loader, device, use_mirror4=True)}
    target = run_dir / "final_test.json"
    target.write_text(json.dumps({"checkpoint_sha256": sha256(checkpoint), "results": outputs}, indent=2) + "\n")
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], required=True)
    parser.add_argument("--run-dir", type=Path, default=Path("/media/8T/PuMengYu/experiments/skin_segmentation/FocalTransNet_ISIC_FRESH100_20261112"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    (train if args.mode == "train" else test)(args.run_dir, torch.device(args.device))


if __name__ == "__main__":
    main()

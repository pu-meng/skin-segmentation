#!/usr/bin/env python3
"""Stage only the fixed PAL ISIC2018 train/validation split from source archives."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

BASE = Path("/home/PuMengYu/experiments/skin_segmentation")
SPLIT = BASE / "data/processed/isic2018_256_dedup_seed42_20261031/split.csv"
ARCHIVES = Path("/media/8T/PuMengYu/Datasets/ISIC2018_archives_20261106")
OUT = Path("/media/8T/PuMengYu/experiments/skin_segmentation/PAL_AUTHOR_ISIC2018_DATA_20261112")
INPUT_ZIP = ARCHIVES / "ISIC2018_Task1-2_Training_Input.zip"
MASK_ZIP = ARCHIVES / "ISIC2018_Task1_Training_GroundTruth.zip"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite existing stage: {OUT}")
    if not SPLIT.is_file() or not INPUT_ZIP.is_file() or not MASK_ZIP.is_file():
        raise SystemExit("Required split or source archive is missing")
    rows = list(csv.DictReader(SPLIT.open(newline="", encoding="utf-8")))
    groups = {"train": [], "val": []}
    for row in rows:
        part = row["split"]
        if part in groups:
            groups[part].append(row["image_id"])
    if (len(groups["train"]), len(groups["val"])) != (1780, 258):
        raise SystemExit(f"Unexpected split counts: { {k: len(v) for k, v in groups.items()} }")
    if len(rows) != 2556:
        raise SystemExit(f"Unexpected total split rows: {len(rows)}")

    OUT.mkdir(parents=True)
    written = {"train": 0, "val": 0}
    with zipfile.ZipFile(INPUT_ZIP) as images, zipfile.ZipFile(MASK_ZIP) as masks:
        for part, ids in groups.items():
            image_dir = OUT / part / "images"
            mask_dir = OUT / part / "masks"
            image_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)
            for image_id in ids:
                image_member = f"ISIC2018_Task1-2_Training_Input/{image_id}.jpg"
                mask_member = f"ISIC2018_Task1_Training_GroundTruth/{image_id}_segmentation.png"
                image_dest = image_dir / f"{image_id}.jpg"
                mask_dest = mask_dir / f"{image_id}.png"  # PAL's ISIC2018 loader expects this name.
                if image_member not in images.NameToInfo or mask_member not in masks.NameToInfo:
                    raise SystemExit(f"Missing source pair: {image_id}")
                with images.open(image_member) as src, image_dest.open("xb") as dst:
                    shutil.copyfileobj(src, dst, length=8 << 20)
                with masks.open(mask_member) as src, mask_dest.open("xb") as dst:
                    shutil.copyfileobj(src, dst, length=8 << 20)
                written[part] += 1

    expected = {"train": 1780, "val": 258}
    actual = {
        part: (len(list((OUT / part / "images").glob("*.jpg"))),
               len(list((OUT / part / "masks").glob("*.png"))))
        for part in groups
    }
    if any(actual[part] != (expected[part], expected[part]) for part in groups):
        raise SystemExit(f"Staged file count mismatch: {actual}")
    manifest = {
        "complete": True,
        "dataset": "ISIC2018",
        "split_csv": str(SPLIT),
        "split_sha256": sha256(SPLIT),
        "archive_inputs": [str(INPUT_ZIP), str(MASK_ZIP)],
        "included_splits_only": ["train", "val"],
        "test_files_staged": False,
        "counts": expected,
        "files": actual,
        "mask_rename": "ISIC_<id>_segmentation.png -> ISIC_<id>.png to match PAL utils/dataset.py",
    }
    (OUT / "stage_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

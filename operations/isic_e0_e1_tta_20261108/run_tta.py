"""Fixed-best four-view TTA for the completed ISIC2018 E0 and E1+E3 runs."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(PROJECT),
    str(PROJECT / "source/comparison"),
    str(PROJECT / "source/bottleneck_mamba"),
    str(PROJECT / "source/e1_e3_combo"),
    str(PROJECT / "source/three_module_impl_20261104"),
    str(PROJECT / "operations/nnunet_mirror_validation"),
]

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import ham_fast_runner as runner
from final_test_pipeline import fixed_best_record, training_ready, validate_outputs
from mirror_core import mirror_logits
from source.bottleneck_mamba.experiment_four_stage_decoder import build_four_stage_baseline
from source.e1_e3_combo.candidate import build_candidate
from source.isic2018 import data as isic

RUN_ROOT = PROJECT / "log/TinyViM_四层_E0_E1E3_ISIC2018_20261105"
PROTOCOL = RUN_ROOT / "protocol.json"
ARMS = {
    "E0": RUN_ROOT / "E0/run",
    "E1_E3": RUN_ROOT / "E1_E3/run",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def build_model(arm: str):
    if arm == "E0":
        return build_four_stage_baseline(pretrained=False)
    return build_candidate(pretrained=False, arm="E1_E3")


class FourViewPredictor(torch.nn.Module):
    """Average inverse-transformed raw logits, then apply sigmoid once."""

    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device
        self.raw_head = None
        self.probability_max_error = 0.0
        self.verify_remaining = 4
        self.forward_seconds = []
        self.hook = model.decoder.head.register_forward_hook(self.capture_head)

    def capture_head(self, _module, _inputs, output):
        self.raw_head = output

    def logits(self, image):
        self.raw_head = None
        probability = self.model(image)
        if self.raw_head is None:
            raise RuntimeError("Decoder output-head hook did not capture logits")
        logits = F.interpolate(
            self.raw_head,
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
        )
        if self.verify_remaining:
            error = float((torch.sigmoid(logits) - probability).abs().max())
            self.probability_max_error = max(self.probability_max_error, error)
            torch.testing.assert_close(torch.sigmoid(logits), probability, rtol=0, atol=0)
            self.verify_remaining -= 1
        return logits

    def forward(self, image):
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        logits = mirror_logits(self.logits, image, enabled=True)
        prediction = torch.sigmoid(logits)
        torch.cuda.synchronize(self.device)
        self.forward_seconds.append(time.perf_counter() - started)
        return prediction

    def close(self):
        self.hook.remove()


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to the authorized physical GPU UUID")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the audited TinyViM selective-scan backend")

    protocol = json.loads(PROTOCOL.read_text())
    config = protocol["data"]
    cache_manifest = isic.verify()
    split_path = Path(config["split_file"])
    if sha256(split_path) != config["split_sha256"]:
        raise ValueError("ISIC2018 split differs from the original training protocol")
    if cache_manifest["counts"] != {"train": 1780, "val": 258, "test": 518}:
        raise ValueError("Unexpected deduplicated ISIC2018 split counts")

    api = runner.baseline_api()
    records = {}
    for arm, run in ARMS.items():
        if not training_ready(run):
            raise RuntimeError(f"{arm}: 300-epoch training/validation artifacts are incomplete")
        fixed = fixed_best_record(run)
        if fixed["history_rows"] != 300 or fixed["split_sha256"] != config["split_sha256"]:
            raise ValueError(f"{arm}: checkpoint/history/split provenance mismatch")
        if (run / "validation/summary.json").is_file() is False:
            raise FileNotFoundError(run / "validation/summary.json")
        output = run.parent / "test/mirror4"
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing TTA output: {output}")
        model = build_model(arm)
        checkpoint = torch.load(fixed["checkpoint"], map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint, strict=True)
        del checkpoint
        if sha256(Path(fixed["checkpoint"])) != fixed["checkpoint_sha256"]:
            raise ValueError(f"{arm}: checkpoint changed during strict loading")
        records[arm] = {"run": str(run), "fixed_best": fixed, "model": model}

    dataset = isic.ISICDataset(isic.CACHE, "test")
    image_ids = {row["image_id"] for row in dataset.rows}
    if len(dataset) != 518 or len(image_ids) != 518 or dataset.training:
        raise ValueError("Expected 518 unique, unaugmented ISIC2018 test images")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(0.15, 0)
    device = torch.device("cuda:0")
    operation_root = PROJECT / "operations/isic_e0_e1_tta_20261108"
    operation_root.mkdir(parents=True, exist_ok=True)
    source_files = [
        Path(__file__),
        PROJECT / "source/bottleneck_mamba/final_test_pipeline.py",
        PROJECT / "operations/nnunet_mirror_validation/mirror_core.py",
        PROJECT / "source/bottleneck_mamba/experiment_four_stage_decoder.py",
        PROJECT / "source/e1_e3_combo/candidate.py",
        PROJECT / "source/isic2018/data.py",
    ]
    manifest = {
        "dataset": "ISIC2018",
        "split": "test",
        "test_images": 518,
        "split_sha256": config["split_sha256"],
        "protocol_sha256": sha256(PROTOCOL),
        "method": "identity, horizontal flip, vertical flip, both flips; inverse-transform and average raw logits; sigmoid once; threshold 0.5",
        "training": "none; existing 300-epoch validation-best checkpoints only",
        "physical_gpu": int(os.environ.get("EXPERIMENT_PHYSICAL_GPU", "1")),
        "gpu_memory_fraction": 0.15,
        "sources_sha256": {str(path): sha256(path) for path in source_files},
        "checkpoints": {
            arm: {
                "path": record["fixed_best"]["checkpoint"],
                "sha256": record["fixed_best"]["checkpoint_sha256"],
                "selected_epoch": record["fixed_best"]["selected_epoch"],
            }
            for arm, record in records.items()
        },
        "started_at": datetime.now().astimezone().isoformat(),
    }
    write_json(operation_root / "manifest.json", manifest)

    for arm, record in records.items():
        model = record["model"].to(device).eval()
        predictor = FourViewPredictor(model, device).eval()
        output = Path(record["run"]).parent / "test/mirror4"
        output.mkdir(parents=True)
        status_path = operation_root / f"{arm}_status.json"
        state = {
            "dataset": "ISIC2018",
            "arm": arm,
            "status": "testing",
            "completed_images": 0,
            "total_images": 518,
            "pid": os.getpid(),
            "test_data_accessed": True,
        }
        write_json(status_path, state)
        started = time.perf_counter()
        criterion = api.BceDiceLoss().to(device)
        result = api.evaluate(predictor, loader, criterion, device, output)
        result.update(
            dataset="ISIC2018",
            split="test",
            model=arm,
            selected_epoch=record["fixed_best"]["selected_epoch"],
            checkpoint=record["fixed_best"]["checkpoint"],
            checkpoint_sha256=record["fixed_best"]["checkpoint_sha256"],
            split_sha256=config["split_sha256"],
            seed=42,
            training_epochs=300,
            tta=True,
            tta_method=manifest["method"],
            test_used_for_checkpoint_selection=False,
            probability_reconstruction_max_abs_error=predictor.probability_max_error,
            forward_seconds_total=sum(predictor.forward_seconds),
            elapsed_seconds=time.perf_counter() - started,
            physical_gpu=manifest["physical_gpu"],
        )
        if predictor.verify_remaining != 0:
            raise RuntimeError(f"{arm}: native probability/logit equivalence check incomplete")
        audit = validate_outputs(output, image_ids, result)
        write_json(output / "summary.json", result)
        write_json(output / "artifact_audit.json", audit)
        (output / "report.txt").write_text(
            f"ISIC2018 {arm}四视图TTA；原训练300轮，固定验证best第{result['selected_epoch']}轮；\n"
            f"独立测试{result['images']}张；Global Dice {100*result['global']['dice']:.4f}%；"
            f"Global IoU {100*result['global']['iou']:.4f}%。\n"
            "对原图及水平、垂直、双翻转图分别推理，逆变换后平均原始logits，再sigmoid并以0.5阈值二值化。\n"
            "本次只补推理，不重新训练，也不使用测试结果选择权重。\n"
        )
        state.update(
            status="complete",
            completed_images=518,
            finished_at=datetime.now().astimezone().isoformat(),
            global_dice=result["global"]["dice"],
            global_iou=result["global"]["iou"],
            audit=audit,
        )
        write_json(status_path, state)
        manifest["checkpoints"][arm]["tta_summary"] = str(output / "summary.json")
        predictor.close()
        del predictor, model
        torch.cuda.empty_cache()
        print(
            f"TTA_COMPLETE {arm} images=518 global_dice={100*result['global']['dice']:.4f}% "
            f"global_iou={100*result['global']['iou']:.4f}%",
            flush=True,
        )

    manifest["finished_at"] = datetime.now().astimezone().isoformat()
    manifest["status"] = "complete"
    write_json(operation_root / "manifest.json", manifest)


if __name__ == "__main__":
    main()

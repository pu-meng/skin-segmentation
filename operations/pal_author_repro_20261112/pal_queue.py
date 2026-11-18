#!/usr/bin/env python3
"""Persistent, fail-closed server-side queue for the PAL ISIC2018 run."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE = Path("/home/PuMengYu/experiments/skin_segmentation")
OP = BASE / "operations/pal_author_repro_20261112"
PYTHON = Path("/home/PuMengYu/anaconda3/envs/medseg/bin/python")
ENTRY = OP / "pal_run.py"
DATA = Path("/media/8T/PuMengYu/experiments/skin_segmentation/PAL_AUTHOR_ISIC2018_DATA_20261112")
ROOT = Path("/media/8T/PuMengYu/experiments/skin_segmentation/PAL_AUTHOR_ISIC2018_FRESH100_20261112")
STATUS = OP / "queue_status.json"
LOCK = OP / "queue.lock"
STOP = OP / "STOP"
FIRST_GATE = 8500
MAX_GATE = 18000
POLL_SECONDS = 60


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def status(state: str, **extra) -> None:
    write_json(STATUS, {
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "state": state, "controller_pid": os.getpid(),
        "dataset": "ISIC2018", "method": "PAL", "gpu_candidates": [0, 1],
        "free_vram_gate_mib": extra.pop("gate_mib", FIRST_GATE),
        "results_root": str(ROOT), **extra,
    })


def gpu_free() -> dict[int, int]:
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True, timeout=15,
    )
    values = {}
    for line in proc.stdout.splitlines():
        idx, free = line.split(",", 1)
        idx, free = int(idx.strip()), int(free.strip())
        if idx in (0, 1):
            values[idx] = free
    return values


def validate_inputs() -> None:
    if not ENTRY.is_file() or not (DATA / "stage_manifest.json").is_file():
        raise RuntimeError("PAL runner or staged data manifest is missing")
    manifest = json.loads((DATA / "stage_manifest.json").read_text())
    if (manifest.get("complete") is not True or manifest.get("test_files_staged") is not False
            or manifest.get("counts") != {"train": 1780, "val": 258}):
        raise RuntimeError("PAL train/validation-only staged data failed manifest validation")
    if ROOT.exists():
        protocol_path = ROOT / "protocol.json"
        status_path = STATUS
        existing_status = json.loads(status_path.read_text()) if status_path.exists() else {}
        prior_attempts = sorted(ROOT.glob("attempt_*"))
        if (not protocol_path.is_file()
                or existing_status.get("state") not in {"waiting_for_gpu", "retrying_after_preflight_oom"}
                or int(existing_status.get("attempt", 0)) != len(prior_attempts)):
            raise RuntimeError(f"Refusing to resume non-waiting PAL result root: {ROOT}")
        protocol = json.loads(protocol_path.read_text())
        if protocol.get("method") != "PAL" or protocol.get("split_sha256") != manifest.get("split_sha256"):
            raise RuntimeError("Existing PAL protocol does not match the staged split")
        for prior in prior_attempts:
            log = prior / "controller_train.log"
            text = log.read_text(errors="replace") if log.is_file() else ""
            if ("PAL_FIRST_OPTIMIZER_STEP_OK" in text or "PAL_TRAINING_COMPLETED" in text
                    or any(prior.rglob("best_model.pth"))):
                raise RuntimeError(f"Refusing automatic restart after a training step: {prior}")


def write_protocol() -> None:
    (ROOT / "protocol.json").write_text(json.dumps({
        "method": "PAL",
        "paper": "PAL: Boosting Skin Lesion Segmentation via Probabilistic Attribute Learning",
        "author_repo": "https://github.com/IsYuchenYuan/PAL",
        "author_commit": "0acce3c52347a70161b44dda26fa90a86a16e4c5",
        "run_scope": "ISIC2018 only",
        "split_csv": str(BASE / "data/processed/isic2018_256_dedup_seed42_20261031/split.csv"),
        "split_sha256": "291dd0995639fa846fc057a8326ec52ad11c83bc167ec4fa1f31d28008e5fba1",
        "train_count": 1780, "validation_count": 258, "test_count": 518,
        "test_files_staged": False,
        "training": {
            "epochs": 100, "batch_size": 16, "input_size": 352,
            "seed": 1337, "backbone": "PVTv2-B2", "optimizer_flag": "Adam",
            "author_code_optimizer": "AdamW", "backbone_lr": 1e-4, "head_lr": 1e-4,
            "weight_decay": 5e-4, "multi_scale_rates": [0.75, 1.0, 1.25],
            "preIter": 1000, "validation_every_steps": 500,
            "checkpoint_selection": "author mIoU on project validation split only",
            "tta": False,
        },
        "architecture": {
            "parameters": 40264627,
            "missing_author_backbone_completed_from": "https://raw.githubusercontent.com/SLDGroup/EMCAD/main/lib/pvtv2.py",
            "pvt_source_sha256": digest(OP / "vendor/PAL/contrastive_model/lib/pvtv2.py"),
            "pvt_weights_sha256": "80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d",
            "weight_keys_expected": 332,
        },
        "compatibility_deviations": [
            "The public PAL snapshot omits its contrastive_model PVTv2 code and PVTv2-B2 weight; the user authorized public PVT completion. The supplemental EMCAD PVTv2 source and PVT-author weight are used; exact identity with the missing PAL file is unknown.",
            "The author code requires NVIDIA Apex O2, unavailable in the shared medseg environment and not installed because other experiments use it. This isolated runner uses the unchanged PAL training loop in FP32; precision differs from the author's O2 run.",
            "A local no-op W&B shim prevents cloud logging; PAL's own wandb=False default is retained.",
            "A public-source typo in argparse keyword `tyep` is bypassed by the isolated launcher; the author checkout is unmodified.",
        ],
        "final_test": "not run during training; after training use frozen validation-selected best, unified raw-probability threshold 0.5 metrics, no TTA, then audit",
        "runner_sha256": digest(ENTRY),
        "staged_data_manifest_sha256": digest(DATA / "stage_manifest.json"),
        "queue_created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }, ensure_ascii=False, indent=2) + "\n")


def run_queue() -> None:
    OP.mkdir(parents=True, exist_ok=True)
    lockf = LOCK.open("a")
    try:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("Another PAL queue controller already holds the lock") from exc

    validate_inputs()
    if not ROOT.exists():
        ROOT.mkdir(parents=False)
        write_protocol()
    status("waiting_for_gpu", gate_mib=FIRST_GATE,
           attempt=len(list(ROOT.glob("attempt_*"))), oom_retries=len(list(ROOT.glob("attempt_*"))))
    gate = FIRST_GATE
    attempted_at_gate: set[tuple[int, int]] = set()
    attempt = len(list(ROOT.glob("attempt_*")))

    while True:
        if STOP.exists():
            status("cancelled_by_stop_file", gate_mib=gate, attempt=attempt)
            return
        try:
            free = gpu_free()
        except Exception as exc:
            status("waiting_gpu_query_error", gate_mib=gate, attempt=attempt, error=repr(exc))
            time.sleep(POLL_SECONDS)
            continue
        eligible = [g for g, mem in free.items()
                    if mem >= gate and (g, gate) not in attempted_at_gate]
        if not eligible:
            if len(attempted_at_gate) == 2:
                if gate >= MAX_GATE:
                    status("blocked_oom_at_max_tested_gate", gate_mib=gate, attempt=attempt,
                           last_free_mib={str(k): v for k, v in free.items()})
                    return
                gate = min(MAX_GATE, gate + 2500)
                attempted_at_gate.clear()
                status("waiting_for_more_free_vram_after_oom", gate_mib=gate, attempt=attempt,
                       last_free_mib={str(k): v for k, v in free.items()})
                continue
            status("waiting_for_gpu", gate_mib=gate, attempt=attempt,
                   free_mib={str(k): v for k, v in free.items()})
            time.sleep(POLL_SECONDS)
            continue

        gpu = max(eligible, key=lambda g: (free[g], -g))
        attempted_at_gate.add((gpu, gate))
        attempt += 1
        attempt_dir = ROOT / f"attempt_{attempt:02d}"
        attempt_dir.mkdir(exist_ok=False)
        log_path = attempt_dir / "controller_train.log"
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": str(gpu), "PAL_PHYSICAL_GPU": str(gpu),
            "PAL_ATTEMPT_DIR": str(attempt_dir), "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "4",
            "PYTHONPATH": os.pathsep.join([
                str(OP / "vendor_deps"), str(OP / "vendor/PAL"),
                str(OP / "vendor/PAL/src"), env.get("PYTHONPATH", ""),
            ]),
        })
        status("preflight_and_training", gate_mib=gate, attempt=attempt, gpu=gpu,
               free_mib_at_start=free[gpu], attempt_dir=str(attempt_dir),
               launch_pid=None)
        with log_path.open("a", buffering=1) as log:
            process = subprocess.Popen(
                [str(PYTHON), "-B", "-u", str(ENTRY)], cwd=str(OP), env=env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            status("preflight_and_training", gate_mib=gate, attempt=attempt, gpu=gpu,
                   free_mib_at_start=free[gpu], attempt_dir=str(attempt_dir),
                   launch_pid=process.pid)
            while process.poll() is None:
                time.sleep(30)
                current = gpu_free()
                status("preflight_and_training", gate_mib=gate, attempt=attempt, gpu=gpu,
                       free_mib_at_start=free[gpu], current_free_mib=current.get(gpu),
                       attempt_dir=str(attempt_dir), launch_pid=process.pid)
        output = log_path.read_text(errors="replace")
        if (process.returncode == 0 and "PAL_PREFLIGHT_OK" in output
                and "PAL_FIRST_OPTIMIZER_STEP_OK" in output
                and "PAL_TRAINING_COMPLETED" in output):
            status("completed", gate_mib=gate, attempt=attempt, gpu=gpu,
                   attempt_dir=str(attempt_dir), launch_pid=process.pid,
                   preflight_passed=True, first_optimizer_step_passed=True)
            return

        oom = "out of memory" in output.lower() or "cuda error: out of memory" in output.lower()
        first_step = "PAL_FIRST_OPTIMIZER_STEP_OK" in output
        if oom and not first_step:
            status("retrying_after_preflight_oom", gate_mib=gate, attempt=attempt, gpu=gpu,
                   attempt_dir=str(attempt_dir), launch_pid=process.pid,
                   preflight_passed="PAL_PREFLIGHT_OK" in output,
                   first_optimizer_step_passed=False)
            continue
        status("failed_or_interrupted", gate_mib=gate, attempt=attempt, gpu=gpu,
               attempt_dir=str(attempt_dir), launch_pid=process.pid,
               exit_code=process.returncode, oom=oom, first_optimizer_step_passed=first_step)
        return


def main() -> None:
    try:
        run_queue()
    except Exception as exc:
        status("controller_error", error=repr(exc))
        raise


if __name__ == "__main__":
    main()

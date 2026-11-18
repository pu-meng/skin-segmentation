"""Resume the grouped-ablation queue after two additional direct starts."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path('/home/PuMengYu/experiments/skin_segmentation')
sys.path.insert(0, str(ROOT))
from operations.ablation_factorial_20261112 import entry

ACTIVE = {
    "a1_swift_only": (1743222, 0),
    "a1_skip_only": (1745714, 1),
    "a1_bottle_decoder_only": (1745783, 0),
    "a1s1_tinyvim_skip_only": (1749470, 0),
    "a1s1_skip_bottle_decoder": (1749523, 1),
    "a1s1_skip_only": (1769768, 0),
    "a1s1_bottle_decoder_only": (1769769, 1),
}


def write(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def alive(pid):
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split()[2]
        return state != "Z"
    except FileNotFoundError:
        return False


def registry(arm, status, gpu=None, reason=None):
    entry._update_registry(arm, status, gpu=gpu, reason=reason)


def main():
    entry.verify()
    run_root = entry.RUN_ROOT
    write(run_root / "queue_status.json", {
        "status": "training_parallel", "active_arms": list(ACTIVE),
        "remaining_arms": [a for a in entry.ARMS if a not in ACTIVE],
        "controller_pid": os.getpid(), "updated_at": time.time(),
    })

    pending = dict(ACTIVE)
    failed = None
    while pending:
        for arm, (pid, gpu) in list(pending.items()):
            if alive(pid):
                continue
            status_path = run_root / arm / "run" / "status.json"
            status = json.loads(status_path.read_text()).get("status") if status_path.exists() else None
            if status != "training_and_validation_complete_test_deferred":
                failed = (arm, f"direct training worker exited; run status={status!r}")
                registry(arm, "failed", gpu=gpu, reason=failed[1])
                break
            pending.pop(arm)
            write(run_root / "queue_status.json", {
                "status": "training_parallel", "completed_training": [a for a in ACTIVE if a not in pending],
                "active_arms": list(pending), "controller_pid": os.getpid(), "updated_at": time.time(),
            })
        if failed:
            break
        time.sleep(10)

    if failed:
        arm, reason = failed
        for remaining in list(entry.ARMS):
            if remaining not in ACTIVE and remaining != arm:
                registry(remaining, "stalled", reason=f"blocked after {arm} failed")
        write(run_root / "queue_status.json", {"status": "failed", "arm": arm, "error": reason, "updated_at": time.time()})
        return

    # Finish the three direct-start arms' fixed-best test deliverables.
    for arm, (_, gpu) in ACTIVE.items():
        try:
            entry._admit_and_run(arm, "test")
            write(run_root / arm / "controller_status.json", {"status": "complete", "arm": arm, "updated_at": time.time()})
            registry(arm, "complete", gpu=gpu)
        except BaseException as error:
            registry(arm, "failed", gpu=gpu, reason=repr(error))
            write(run_root / "queue_status.json", {"status": "failed", "arm": arm, "error": repr(error), "updated_at": time.time()})
            return

    # Continue all not-yet-started cells serially, still without a preflight.
    for arm in entry.ARMS:
        if arm in ACTIVE:
            continue
        try:
            entry._admit_and_run(arm, "train_direct")
            entry._admit_and_run(arm, "test")
            write(run_root / arm / "controller_status.json", {"status": "complete", "arm": arm, "updated_at": time.time()})
            registry(arm, "complete")
        except BaseException as error:
            registry(arm, "failed", reason=repr(error))
            for remaining in list(entry.ARMS)[list(entry.ARMS).index(arm) + 1:]:
                if remaining not in ACTIVE:
                    registry(remaining, "stalled", reason=f"blocked after {arm} failed")
            write(run_root / "queue_status.json", {"status": "failed", "arm": arm, "error": repr(error), "updated_at": time.time()})
            return
    write(run_root / "queue_status.json", {"status": "complete", "arms": list(entry.ARMS), "updated_at": time.time()})


if __name__ == "__main__":
    main()

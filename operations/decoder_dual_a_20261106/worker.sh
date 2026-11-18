#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/PuMengYu/experiments/skin_segmentation
RUN=/media/8T/PuMengYu/experiments/skin_segmentation/DecoderDualA_SCREEN100of300_20261106
PY=/home/PuMengYu/anaconda3/envs/medseg/bin/python
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
printf 'preflight\n' > "$RUN/controller.phase"
touch "$RUN/PREFLIGHT_GO"
# Single candidate: pass marker is written only after the real acceptance below.
timeout --foreground 3600 "$PY" -u -m source.decoder_dual_a_20261106.entry preflight 2>&1 | tee "$RUN/preflight.stdout.log"
"$PY" -c 'import json;from pathlib import Path;p=Path("/media/8T/PuMengYu/experiments/skin_segmentation/DecoderDualA_SCREEN100of300_20261106");a=json.loads((p/"DualA/preflight/preflight.json").read_text());assert a["passed"] and len(a["steps"])==3;print("PREFLIGHT_VERIFIED: weights discarded; fresh process next",flush=True)'
printf 'train\n' > "$RUN/controller.phase"
touch "$RUN/PREFLIGHT_ALL_PASSED" "$RUN/TRAIN_GO"
remaining=$(( DEADLINE - $(date +%s) ))
(( remaining > 0 ))
timeout --foreground "$remaining" "$PY" -u -m source.decoder_dual_a_20261106.entry train 2>&1 | tee "$RUN/train.stdout.log"
"$PY" -c 'import json;from pathlib import Path;p=Path("/media/8T/PuMengYu/experiments/skin_segmentation/DecoderDualA_SCREEN100of300_20261106/DualA/train");s=json.loads((p/"status.json").read_text());assert s["status"]=="paused_at_screen_boundary" and s["epoch_completed"]==100 and s["optimizer_updates"]==87300 and not s["automatic_continuation"] and not s["test_data_accessed"];print("SCREEN100_PAUSED_VERIFIED_NO_TEST",flush=True)'
printf 'paused_at_100\n' > "$RUN/controller.phase"

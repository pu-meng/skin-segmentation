#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/PuMengYu/experiments/skin_segmentation
PY=/home/PuMengYu/anaconda3/envs/medseg/bin/python
GPU_UUID=GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b
GPU1_TRAIN_PID=2989499
S3H_RUN=/media/8T/PuMengYu/experiments/skin_segmentation/S3HNet_HAM_FRESH200_20261114
FOCAL_RUN=/media/8T/PuMengYu/experiments/skin_segmentation/FocalTransNet_ISIC_FRESH100_GPU1_RESUME_FROMBEST44_20261114
OPS="$ROOT/operations/focaltransnet_isic_20261114"
STATE="$OPS/gpu1_recovery_queue_status.json"
LOG="$OPS/gpu1_recovery_queue.log"
QUEUE_STARTED=$(date +%s)

cd "$ROOT"
export EXPERIMENT_PHYSICAL_GPU=1 EXPERIMENT_GPU_UUID="$GPU_UUID"
export CUDA_VISIBLE_DEVICES="$GPU_UUID" MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1

write_state() {
  STATE_STATUS="$1" STATE_DETAIL="$2" STATE_FILE="$STATE" "$PY" - <<'PY'
import json, os, time
from pathlib import Path
Path(os.environ['STATE_FILE']).write_text(json.dumps({
    'status': os.environ['STATE_STATUS'], 'detail': os.environ['STATE_DETAIL'],
    'updated_at': time.time(), 'gpu': 1, 'supervisor_pid': os.getppid(),
}, ensure_ascii=False, indent=2) + '\n')
PY
}

wait_for_gpu1() {
  while true; do
    FREE_MIB=$(nvidia-smi --query-gpu=uuid,memory.free --format=csv,noheader | awk -F, -v uuid="$GPU_UUID" '$1 == uuid {gsub(/ MiB/, "", $2); gsub(/ /, "", $2); print $2}')
    if [[ -n "$FREE_MIB" && "$FREE_MIB" -ge 4096 ]]; then
      return
    fi
    write_state waiting_for_gpu "GPU1 free memory: ${FREE_MIB:-unknown} MiB; need 4096 MiB"
    sleep 30
  done
}

mkdir -p "$OPS"
exec >> "$LOG" 2>&1
write_state waiting_for_s3h "Waiting for S3H-Net resume PID $GPU1_TRAIN_PID"
while kill -0 "$GPU1_TRAIN_PID" 2>/dev/null; do sleep 30; done

if [[ ! -f "$S3H_RUN/train_complete.json" ]]; then
  write_state blocked_s3h_incomplete "S3H training process exited without train_complete.json; downstream GPU1 jobs were not started"
  exit 1
fi
S3H_STATUS=$("$PY" -c "import json;print(json.load(open('$S3H_RUN/status.json')).get('status',''))")
S3H_UPDATED=$("$PY" -c "import json;print(json.load(open('$S3H_RUN/status.json')).get('updated_at',0))")
if [[ "$S3H_STATUS" != training_complete_test_deferred ]] || ! "$PY" -c "import sys;sys.exit(float('$S3H_UPDATED') >= $QUEUE_STARTED)"; then
  write_state blocked_s3h_status "S3H completion status is stale or incomplete; downstream GPU1 jobs were not started"
  exit 1
fi

wait_for_gpu1
write_state testing_s3h "Running S3H-Net HAM none/mirror4 final test"
"$PY" -B -u operations/s3hnet_ham_20261114/run.py --mode test --run-dir "$S3H_RUN"

wait_for_gpu1
write_state training_focal_preflight "Checking FocalTransNet batch on GPU1 from saved best epoch 44"
"$PY" -B -u operations/focaltransnet_isic_20261114/resume_from_best.py cuda:0 --preflight

wait_for_gpu1
write_state training_focal "Continuing FocalTransNet from saved best epoch 44 through epoch 100"
"$PY" -B -u operations/focaltransnet_isic_20261114/resume_from_best.py cuda:0

wait_for_gpu1
write_state testing_focal "Running FocalTransNet ISIC none/mirror4 final test"
"$PY" -B -u operations/focaltransnet_isic_20261114/run.py --mode test --run-dir "$FOCAL_RUN" --device cuda:0
write_state complete "S3H-Net HAM and FocalTransNet ISIC training/test queue completed"

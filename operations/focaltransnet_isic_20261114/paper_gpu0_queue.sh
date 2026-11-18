#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/PuMengYu/experiments/skin_segmentation
OPS="$ROOT/operations/focaltransnet_isic_20261114"
SEED_OPS="$ROOT/operations/a1_esb_seed_matrix_20261114"
PY=/home/PuMengYu/anaconda3/envs/medseg/bin/python
GPU0_UUID=GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde
GPU1_UUID=GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b
S3H_RUN=/media/8T/PuMengYu/experiments/skin_segmentation/S3HNet_HAM_FRESH200_20261114
FOCAL_RUN=/media/8T/PuMengYu/experiments/skin_segmentation/FocalTransNet_ISIC_FRESH100_GPU0_RESUME_FROMBEST44_20261114
S3H_PID=2989499

write_state() {
  STATE_FILE="$1" STATE_STATUS="$2" STATE_DETAIL="$3" STATE_PID="${4:-}" "$PY" - <<'PY'
import json, os, time
from pathlib import Path
d={'status':os.environ['STATE_STATUS'],'detail':os.environ['STATE_DETAIL'],
   'updated_at':time.time()}
if os.environ.get('STATE_PID'): d['pid']=int(os.environ['STATE_PID'])
Path(os.environ['STATE_FILE']).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
PY
}

wait_gpu() {
  local uuid="$1" need="$2"
  while true; do
    local free
    free=$(nvidia-smi --query-gpu=uuid,memory.free --format=csv,noheader | awk -F, -v u="$uuid" '$1==u {gsub(/ MiB/,"",$2);gsub(/ /,"",$2);print $2}')
    if [[ -n "$free" && "$free" -ge "$need" ]]; then return; fi
    sleep 30
  done
}

run_s3h_test() {
  local status_file="$OPS/s3h_test_queue_status.json"
  write_state "$status_file" waiting_for_s3h "Wait for S3H HAM training to finish on GPU1" "$S3H_PID"
  while true; do
    if [[ -f "$S3H_RUN/failure.json" ]]; then
      write_state "$status_file" failed "S3H training failure is recorded; final test was not started"
      return 1
    fi
    if [[ -f "$S3H_RUN/train_complete.json" ]]; then break; fi
    sleep 30
  done
  while true; do
    local proc_state
    proc_state=$(ps -p "$S3H_PID" -o stat= 2>/dev/null | awk 'NF {print substr($1,1,1)}')
    [[ -z "$proc_state" || "$proc_state" == Z ]] && break
    sleep 10
  done
  if [[ ! -f "$S3H_RUN/test_complete.json" ]]; then
    wait_gpu "$GPU1_UUID" 4096
    CUDA_VISIBLE_DEVICES="$GPU1_UUID" EXPERIMENT_PHYSICAL_GPU=1 EXPERIMENT_GPU_UUID="$GPU1_UUID" \
      MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
      "$PY" -B -u "$ROOT/operations/s3hnet_ham_20261114/run.py" --mode test --run-dir "$S3H_RUN"
  fi
  "$PY" "$ROOT/operations/expctl.py" update --id S3HNET_HAM_FRESH200_20261114 \
    --status complete --note 'HAM resume training and fixed-split none/mirror4 test complete; verify test_complete.json and per-mode audits.'
  write_state "$status_file" complete "S3H HAM none/mirror4 test complete"
}

run_focal_gpu0() {
  local status_file="$OPS/focal_gpu0_queue_status.json"
  write_state "$status_file" waiting_for_seed_matrix "Wait for E+S+B seeds 40/41/43 training and TTA4 tests to complete"
  while true; do
    local matrix_status
    matrix_status=$("$PY" -c "import json;print(json.load(open('$SEED_OPS/matrix_status.json')).get('status',''))")
    if [[ "$matrix_status" == failed ]]; then
      write_state "$status_file" blocked "E+S+B seed matrix failed; FocalTransNet was not started"
      return 1
    fi
    [[ "$matrix_status" == complete ]] && break
    sleep 30
  done

  # GPU0 capacity is user-confirmed; do not gate on a free-memory threshold
  # or run a separate dummy batch before the actual continuation.
  CUDA_VISIBLE_DEVICES="$GPU0_UUID" EXPERIMENT_PHYSICAL_GPU=0 EXPERIMENT_GPU_UUID="$GPU0_UUID" \
    MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
    "$PY" -B -u "$OPS/resume_from_best.py" cuda:0 > "$OPS/focal_gpu0_train.log" 2>&1 &
  local train_pid=$!
  write_state "$status_file" training "Continuing FocalTransNet ISIC from best epoch44 through epoch100 on GPU0" "$train_pid"
  "$PY" "$ROOT/operations/expctl.py" update --id FOCALTRANSNET_ISIC_GPU0_RESUME_FROMBEST44_20261114 \
    --status training --gpu 0 --pid "$train_pid" --note 'Direct continuation from saved best epoch44, training through epoch100 on GPU0; no free-memory threshold or dummy batch preflight.'
  wait "$train_pid"

  CUDA_VISIBLE_DEVICES="$GPU0_UUID" EXPERIMENT_PHYSICAL_GPU=0 EXPERIMENT_GPU_UUID="$GPU0_UUID" \
    MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
    "$PY" -B -u "$ROOT/operations/focaltransnet_isic_20261114/run.py" \
    --mode test --run-dir "$FOCAL_RUN" --device cuda:0 > "$OPS/focal_gpu0_test.log" 2>&1 &
  local test_pid=$!
  write_state "$status_file" testing "FocalTransNet ISIC fixed-checkpoint none/mirror4 test on GPU0" "$test_pid"
  "$PY" "$ROOT/operations/expctl.py" update --id FOCALTRANSNET_ISIC_GPU0_RESUME_FROMBEST44_20261114 \
    --status testing --gpu 0 --pid "$test_pid" --note 'Epoch100 continuation complete; final none/mirror4 test is running on GPU0.'
  wait "$test_pid"
  "$PY" "$ROOT/operations/expctl.py" update --id FOCALTRANSNET_ISIC_GPU0_RESUME_FROMBEST44_20261114 \
    --status complete --epochs-done 100 --epochs 100 --gpu 0 --note 'Continuation and ISIC none/mirror4 final test complete; see final_test.json.'
  write_state "$status_file" complete "FocalTransNet ISIC continuation and final test complete"
}

mkdir -p "$OPS"
exec >> "$OPS/paper_gpu0_queue.log" 2>&1
run_s3h_test &
S3H_TEST_WORKER=$!
run_focal_gpu0 &
FOCAL_WORKER=$!
wait "$S3H_TEST_WORKER"
wait "$FOCAL_WORKER"

#!/bin/bash
set -euo pipefail
cd /home/PuMengYu/experiments/skin_segmentation
RUN='/home/PuMengYu/experiments/skin_segmentation/log/TinyViM_E123_CandidateA_E2Spectral_B3x3_HAM100_20261106'
exec 9>"$RUN/controller.lock"
flock -n 9 || exit 73
export CUDA_VISIBLE_DEVICES=1 EXPERIMENT_PHYSICAL_GPU=1 EXPERIMENT_GPU_UUID=GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/home/PuMengYu/anaconda3/envs/medseg/bin/python
echo $$ > "$RUN/controller.pid"
phase=starting
finish() {
 rc=$?
 printf '%s\n' "$rc" > "$RUN/controller.exitcode"
 printf '{"phase":"%s","exitcode":%d,"finished_at":"%s"}\n' "$phase" "$rc" "$(date -Is)" > "$RUN/controller_status.json"
}
trap finish EXIT
printf '{"phase":"preflight","deadline":"72h maximum; failure blocks training","controller_pid":%d}\n' $$ > "$RUN/controller_status.json"
phase=preflight
timeout --signal=TERM 3600 "$PY" -u -m source.e123_candidate_a_20261106.entry preflight 2>&1 | tee "$RUN/preflight_console.log"
echo 0 > "$RUN/preflight.exitcode"
phase=train
printf '{"phase":"train","budget":100,"test":false,"controller_pid":%d}\n' $$ > "$RUN/controller_status.json"
timeout --signal=TERM 259200 "$PY" -u -m source.e123_candidate_a_20261106.entry train 2>&1 | tee "$RUN/train_console.log"
echo 0 > "$RUN/train.exitcode"
phase=complete_validation_only

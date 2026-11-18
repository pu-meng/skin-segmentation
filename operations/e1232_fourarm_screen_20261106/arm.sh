#!/bin/bash
set -euo pipefail
cd /home/PuMengYu/experiments/skin_segmentation
ARM="$1"
RUN=/home/PuMengYu/experiments/skin_segmentation/log/TinyViM_E1232_FourArm_SCREEN100of300_20261106
exec 9>"$RUN/$ARM/controller.lock"
flock -n 9 || exit 73
export CUDA_VISIBLE_DEVICES=GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde
export EXPERIMENT_PHYSICAL_GPU=0 EXPERIMENT_GPU_UUID=GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/home/PuMengYu/anaconda3/envs/medseg/bin/python
printf '%s\n' $$ > "$RUN/$ARM/controller.pid"
phase=preflight
finish() { rc=$?; printf '%s\n' "$rc" > "$RUN/$ARM/controller.exitcode"; printf '{"phase":"%s","exitcode":%d}\n' "$phase" "$rc" > "$RUN/$ARM/controller_status.json"; }
trap finish EXIT
printf '{"phase":"preflight"}\n' > "$RUN/$ARM/controller_status.json"
timeout --signal=TERM 3600 "$PY" -u -m source.e1232_fourarm_screen_20261106.entry preflight "$ARM" 2>&1 | tee "$RUN/$ARM/preflight_console.log"
echo 0 > "$RUN/$ARM/preflight.exitcode"
phase=train
printf '{"phase":"train"}\n' > "$RUN/$ARM/controller_status.json"
timeout --signal=TERM 259200 "$PY" -u -m source.e1232_fourarm_screen_20261106.entry train "$ARM" 2>&1 | tee "$RUN/$ARM/train_console.log"
echo 0 > "$RUN/$ARM/train.exitcode"
phase=paused100_validation_only

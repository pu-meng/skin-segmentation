#!/bin/bash
set -euo pipefail
cd /home/PuMengYu/experiments/skin_segmentation
exec 9>/home/PuMengYu/experiments/skin_segmentation/operations/resume300_20261106/R3/controller.lock
flock -n 9 || exit 73
export RESUME_ARM=R3 CUDA_VISIBLE_DEVICES=GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde EXPERIMENT_PHYSICAL_GPU=0
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
PY=/home/PuMengYu/anaconda3/envs/medseg/bin/python
printf '%s\n' $$ > /home/PuMengYu/experiments/skin_segmentation/operations/resume300_20261106/R3/controller.pid
phase=preflight
finish() { rc=$?; printf '%s\n' "$rc" > /home/PuMengYu/experiments/skin_segmentation/operations/resume300_20261106/R3/controller.exitcode; printf '{"phase":"%s","exitcode":%d}\n' "$phase" "$rc" > /home/PuMengYu/experiments/skin_segmentation/operations/resume300_20261106/R3/controller_status.json; }
trap finish EXIT
for phase in preflight train test; do
 printf '{"phase":"%s"}\n' "$phase" > /home/PuMengYu/experiments/skin_segmentation/operations/resume300_20261106/R3/controller_status.json
 timeout --foreground --signal=TERM 259200 "$PY" -u /home/PuMengYu/experiments/skin_segmentation/operations/resume300_20261106/entry.py "$phase" 2>&1 | tee /home/PuMengYu/experiments/skin_segmentation/operations/resume300_20261106/R3/"$phase".log
 echo 0 > /home/PuMengYu/experiments/skin_segmentation/operations/resume300_20261106/R3/"$phase".exitcode
done
phase=complete

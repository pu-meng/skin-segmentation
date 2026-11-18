#!/bin/bash
set -euo pipefail
cd /home/PuMengYu/experiments/skin_segmentation
exec 9>/home/PuMengYu/experiments/skin_segmentation/operations/e1232_matrix_20261106/BT/controller.lock
flock -n 9 || exit 73
export MATRIX_ARM=BT CUDA_VISIBLE_DEVICES=GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b EXPERIMENT_PHYSICAL_GPU=1
export MATRIX_COHORT="BT BS AU"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
PY=/home/PuMengYu/anaconda3/envs/medseg/bin/python
printf '%s\n' $$ > /home/PuMengYu/experiments/skin_segmentation/operations/e1232_matrix_20261106/BT/controller.pid
phase=waiting_resources
finish() { rc=$?; printf '%s\n' "$rc" > /home/PuMengYu/experiments/skin_segmentation/operations/e1232_matrix_20261106/BT/controller.exitcode; printf '{"phase":"%s","exitcode":%d}\n' "$phase" "$rc" > /home/PuMengYu/experiments/skin_segmentation/operations/e1232_matrix_20261106/BT/controller_status.json; }
trap finish EXIT
"$PY" -u /home/PuMengYu/experiments/skin_segmentation/operations/e1232_matrix_20261106/admission.py 2>&1 | tee /home/PuMengYu/experiments/skin_segmentation/operations/e1232_matrix_20261106/BT/admission.log
for phase in preflight train test; do
 if [ "$phase" = train ]; then
  for peer in ${MATRIX_COHORT}; do
   until [ -f /home/PuMengYu/experiments/skin_segmentation/operations/e1232_matrix_20261106/$peer/preflight.exitcode ]; do
    [ ! -f /home/PuMengYu/experiments/skin_segmentation/operations/e1232_matrix_20261106/$peer/failure.json ] || exit 75
   sleep 2
   done
  done
 fi
 printf '{"phase":"%s"}\n' "$phase" > /home/PuMengYu/experiments/skin_segmentation/operations/e1232_matrix_20261106/BT/controller_status.json
 timeout --foreground --signal=TERM 432000 "$PY" -u /home/PuMengYu/experiments/skin_segmentation/operations/e1232_matrix_20261106/worker.py "$phase" 2>&1 | tee /home/PuMengYu/experiments/skin_segmentation/operations/e1232_matrix_20261106/BT/"$phase".log
 echo 0 > /home/PuMengYu/experiments/skin_segmentation/operations/e1232_matrix_20261106/BT/"$phase".exitcode
done
phase=complete

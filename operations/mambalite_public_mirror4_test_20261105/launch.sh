#!/bin/bash
set -uo pipefail
cd /home/PuMengYu/experiments/skin_segmentation
O=operations/mambalite_public_mirror4_test_20261105
OUT=log/mambalite_public_mirror4_test_20261105
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
/home/PuMengYu/anaconda3/envs/medseg/bin/python -u "$O/run.py" run 2>&1 | tee "$OUT/stdout.log"
e=${PIPESTATUS[0]}
printf '%s\n' "$e" > "$OUT/exitcode"
printf '%s\n' "$e" > "$O/exitcode"
exit "$e"

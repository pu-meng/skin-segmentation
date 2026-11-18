#!/bin/bash
set -uo pipefail
cd /home/PuMengYu/experiments/skin_segmentation || exit 90
OUT=/home/PuMengYu/experiments/skin_segmentation/log/mambalite_fig2b_mirror4_test_20261105
export CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
/home/PuMengYu/anaconda3/envs/medseg/bin/python -B -u operations/mambalite_fig2b_mirror4_test_20261105/run.py run 2>&1 | tee "$OUT/stdout.log"
code=${PIPESTATUS[0]}
printf '%s\n' "$code" > "$OUT/exitcode"
printf 'EVALUATION EXITCODE %s\n' "$code"
exit "$code"

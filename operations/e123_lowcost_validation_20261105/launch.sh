#!/usr/bin/env bash
set -uo pipefail
cd /home/PuMengYu/experiments/skin_segmentation
OUT='/home/PuMengYu/experiments/skin_segmentation/log/E123.1_低成本验证核查_20261105'
export CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
/home/PuMengYu/anaconda3/envs/medseg/bin/python -B -u operations/e123_lowcost_validation_20261105/run.py run 2>&1 | tee "$OUT/stdout.log"
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > "$OUT/exitcode"
printf 'EXITCODE=%s\n' "$rc"
exit "$rc"

#!/usr/bin/env bash
set -uo pipefail
cd /home/PuMengYu/experiments/skin_segmentation
OPS=/home/PuMengYu/experiments/skin_segmentation/operations/new_completed_bypass_20261106
OUT=/media/8T/PuMengYu/diagnostics/new_completed_bypass_20261106
export CUDA_VISIBLE_DEVICES=1 BD_ARM=D4-B3D PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
/home/PuMengYu/anaconda3/envs/medseg/bin/python -B -u "$OPS/run.py" run 2>&1 | tee "$OUT/stdout.log"
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > "$OUT/inference.exitcode"
if [ "$rc" -eq 0 ]; then
 CUDA_VISIBLE_DEVICES='' /home/PuMengYu/anaconda3/envs/medseg/bin/python -B -u "$OPS/audit.py" 2>&1 | tee "$OUT/audit.log"
 rc=${PIPESTATUS[0]}
fi
printf '%s\n' "$rc" > "$OUT/exitcode"
printf 'EXITCODE=%s\n' "$rc"
exit "$rc"

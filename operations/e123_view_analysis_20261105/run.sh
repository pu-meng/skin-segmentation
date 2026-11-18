#!/usr/bin/env bash
set -euo pipefail
cd /home/PuMengYu/experiments/skin_segmentation
HERE="$PWD/operations/e123_view_analysis_20261105"
exec 9>"$HERE/controller.lock"
flock -n 9 || exit 91
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MKL_THREADING_LAYER=GNU
export CUDA_VISIBLE_DEVICES=GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b
set +e
timeout 900 /home/PuMengYu/anaconda3/envs/medseg/bin/python -B -u "$HERE/diagnose.py" 2>&1 | tee "$HERE/console.log"
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > "$HERE/exitcode"
exit "$rc"

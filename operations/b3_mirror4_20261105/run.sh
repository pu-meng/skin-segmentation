#!/usr/bin/env bash
set -euo pipefail
cd /home/PuMengYu/experiments/skin_segmentation
OUT="$PWD/log/TinyViM_基础架构四对照_HAM_20261105/D4-B3/mirror4_test"
exec 9>"$OUT/controller.lock"
flock -n 9 || exit 91
export BD_ARM=D4-B3 CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
set +e
/home/PuMengYu/anaconda3/envs/medseg/bin/python -B -u operations/b3_mirror4_20261105/run_test.py run 2>&1 | tee "$OUT/console.log"
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > "$OUT/exitcode"
exit "$rc"

#!/usr/bin/env bash
set -euo pipefail
cd /home/PuMengYu/experiments/skin_segmentation
HERE=/home/PuMengYu/experiments/skin_segmentation/operations/e123_early_20261105
exec 9>"$HERE/controller.lock"
flock -n 9
export CUDA_VISIBLE_DEVICES=GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b
export EXPERIMENT_PHYSICAL_GPU=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MKL_THREADING_LAYER=GNU
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=index,uuid,memory.free,utilization.gpu --format=csv,noheader
set +e
/home/PuMengYu/anaconda3/envs/medseg/bin/python -B -u "$HERE/early_test.py" 2>&1 | tee "$HERE/console.log"
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > "$HERE/exitcode"
exit "$rc"

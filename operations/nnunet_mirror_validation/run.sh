#!/usr/bin/env bash
set -euo pipefail
cd /home/PuMengYu/experiments/skin_segmentation
OUT="$PWD/log/nnunet_mirror_validation_20261105"
export CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
set +e
/home/PuMengYu/anaconda3/envs/medseg/bin/python -u operations/nnunet_mirror_validation/run_validation.py run 2>&1 | tee "$OUT/console.log"
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > "$OUT/exitcode"
exit "$rc"

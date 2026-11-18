#!/bin/bash
set -uo pipefail
cd /home/PuMengYu/experiments/skin_segmentation/operations/engineering_speed_lab
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MKL_THREADING_LAYER=GNU
flock -n probe.lock timeout --signal=TERM --kill-after=10 600 /home/PuMengYu/anaconda3/envs/medseg/bin/python -B -u triton_probe.py 2>&1 | tee probe_console.log
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > probe.exitcode
exit "$rc"

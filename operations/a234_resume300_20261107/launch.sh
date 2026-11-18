#!/bin/bash
set -euo pipefail
cd /home/PuMengYu/experiments/skin_segmentation
OPS=/home/PuMengYu/experiments/skin_segmentation/operations/a234_resume300_20261107
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
printf '%s\n' $$ > "$OPS/launcher.pid"
finish() { rc=$?; printf '%s\n' "$rc" > "$OPS/controller.exitcode"; printf '{"exitcode":%d,"finished_at":"%s"}\n' "$rc" "$(date -Is)" > "$OPS/launcher_exit.json"; }
trap finish EXIT
for a in A2 A3 A4; do test -f "$OPS/$a/cpu_roundtrip.json"; done
# The deadline also bounds active preflight/train/test, not just resource waiting.
timeout --foreground --signal=TERM 259200 /home/PuMengYu/anaconda3/envs/medseg/bin/python -B -u "$OPS/entry.py" controller 2>&1 | tee "$OPS/controller.log"

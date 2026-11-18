#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/PuMengYu/experiments/skin_segmentation
OPS="$ROOT/operations/a1_esb_seed_matrix_20261114"
PY=/home/PuMengYu/anaconda3/envs/medseg/bin/python
while true; do
  STATUS=$("$PY" -c "import json;print(json.load(open('$OPS/matrix_status.json')).get('status',''))")
  case "$STATUS" in
    complete)
      exec "$PY" -B -u "$OPS/finalize_report.py" >> "$OPS/finalize_report.log" 2>&1
      ;;
    failed)
      exit 1
      ;;
  esac
  sleep 60
done

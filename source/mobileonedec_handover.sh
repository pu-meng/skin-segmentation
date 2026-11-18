#!/bin/bash
# MobileOneDec: ISIC2018 was started with `controller ISIC2018 0`.
# The vendor entry point also has serial_queue() (ISIC2018 -> HAM10000 on GPU0, skipping
# any dataset whose completion.json says complete). Starting serial_queue now would collide
# on the controller.lock, so wait for the running ISIC controller to finish, then hand over.
set -u

PY=/home/PuMengYu/anaconda3/envs/medseg/bin/python
OPS=/home/PuMengYu/experiments/skin_segmentation/operations/a1_s1_skipv06_mobileonedec_20261110
RUN=/media/8T/PuMengYu/experiments/skin_segmentation/A1_S1_SKIPV06_MOBILEONEDEC_HAM_ISIC_FRESH300_20261110

while true; do
  st=$("$PY" -c "import json;print(json.load(open('$RUN/ISIC2018/controller_status.json'))['status'])" 2>/dev/null || echo none)
  if [ "$st" = "complete" ]; then
    echo "$(date -Is) ISIC controller complete, handing over to serial_queue" 
    break
  fi
  sleep 120
done

cd "$OPS" || exit 1
exec "$PY" -B -u skip_entry.py serial_queue

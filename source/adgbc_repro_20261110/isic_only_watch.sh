#!/bin/bash
# AD-GBC: run only ISIC2018. The user stopped the HAM10000 stage; ISIC training continues.
# This watcher waits for the running ISIC trainer to finish and then runs the final test
# in our metric convention. It replaces the HAM stages that run_queue.sh would have started.
set -u

PY=/home/PuMengYu/anaconda3/envs/medseg/bin/python
SRC=/home/PuMengYu/experiments/skin_segmentation/source/adgbc_repro_20261110
OUT=/media/8T/PuMengYu/experiments/skin_segmentation
RUN=$OUT/ADGBC_L_ISIC2018_20261110

cd "$SRC" || exit 1
export PYTHONPATH="$SRC"

while true; do
  st=$("$PY" -c "import json;print(json.load(open('$RUN/status.json')).get('status',''))" 2>/dev/null || echo none)
  case "$st" in
    complete) echo "$(date -Is) ADGBC ISIC train complete -> final test"; break ;;
    failed)   echo "$(date -Is) ADGBC ISIC train FAILED; not evaluating"; exit 1 ;;
    *)        sleep 120 ;;
  esac
done

"$PY" -u evaluate_final.py --dataset ISIC2018 --out "$RUN" --gpu 0
echo "$(date -Is) ADGBC ISIC final test rc=$?"

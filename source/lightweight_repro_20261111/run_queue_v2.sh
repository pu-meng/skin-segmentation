#!/bin/bash
# Lightweight-baseline queue, v2.
# FuseUNet-ISIC is run manually on GPU0, so this queue no longer starts it: it waits for
# the external run to finish, then evaluates. CMUNeXt-ISIC was started by v1 of this queue
# and is still training on GPU1; same treatment.
# After the two ISIC runs are scored, the two HAM runs are launched when a slot frees up.
set -u

PY=/home/PuMengYu/anaconda3/envs/medseg/bin/python
SRC=/home/PuMengYu/experiments/skin_segmentation/source/lightweight_repro_20261111
OUT=/media/8T/PuMengYu/experiments/skin_segmentation
GPU=${SWAP_GPU:-1}
NEED_FREE=5000
LOG=$OUT/LIGHTWEIGHT_QUEUE_V2_$(date +%Y%m%d_%H%M).log

cd "$SRC" || exit 1
export PYTHONPATH="$SRC"
export CUDA_VISIBLE_DEVICES="$GPU"

echo "=== lightweight queue v2 start $(date -Is) gpu=$GPU ===" | tee -a "$LOG"

status_of() {
  local dir=$1
  [ -f "$OUT/$dir/status.json" ] || { echo none; return; }
  "$PY" -c "import json;print(json.load(open('$OUT/$dir/status.json')).get('status',''))" 2>/dev/null || echo none
}

await_train() {
  local dir=$1
  while true; do
    local st
    st=$(status_of "$dir")
    case "$st" in
      complete) echo "[await] $dir complete $(date -Is)" | tee -a "$LOG"; return 0 ;;
      failed)   echo "[await] $dir FAILED $(date -Is)" | tee -a "$LOG"; return 1 ;;
      none)     echo "[await] $dir has no status yet $(date -Is)" >> "$LOG"; sleep 120 ;;
      *)        echo "[await] $dir status=$st $(date -Is)" >> "$LOG"; sleep 180 ;;
    esac
  done
}

eval_stage() {
  local dataset=$1 dir=$2
  echo "--- final test $dir ---" | tee -a "$LOG"
  "$PY" -u evaluate_final.py --dataset "$dataset" --out "$OUT/$dir" --gpu 0 --workers 2 >> "$LOG" 2>&1
  echo "--- final test done $dir rc=$? $(date -Is) ---" | tee -a "$LOG"
}

wait_gpu() {
  while true; do
    local free procs
    free=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
    procs=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)
    if [ "$procs" -lt 4 ] && [ "$free" -ge "$NEED_FREE" ]; then
      echo "[queue] gpu$GPU ready free=${free}MiB procs=$procs $(date -Is)" | tee -a "$LOG"
      return 0
    fi
    echo "[queue] waiting gpu$GPU free=${free}MiB procs=$procs $(date -Is)" >> "$LOG"
    sleep 180
  done
}

train_stage() {
  local model=$1 dataset=$2 dir=$3
  wait_gpu
  echo "--- train $dir ---" | tee -a "$LOG"
  "$PY" -u train_repro.py --model "$model" --dataset "$dataset" \
      --out "$OUT/$dir" --gpu 0 --epochs 300 --batch_size 8 --workers 4 >> "$LOG" 2>&1
  if [ $? -ne 0 ]; then
    echo "!!! train failed $dir $(date -Is)" | tee -a "$LOG"
    return 1
  fi
  eval_stage "$dataset" "$dir"
}

await_train CMUNEXT_ISIC2018_20261111 && eval_stage ISIC2018 CMUNEXT_ISIC2018_20261111
await_train FUSEUNET_ISIC2018_20261111 && eval_stage ISIC2018 FUSEUNET_ISIC2018_20261111
train_stage cmunext  HAM10000 CMUNEXT_HAM10000_20261111
train_stage fuseunet HAM10000 FUSEUNET_HAM10000_20261111

echo "=== lightweight queue v2 done $(date -Is) ===" | tee -a "$LOG"

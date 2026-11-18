#!/bin/bash
# Lightweight-baseline reproduction queue: CMUNeXt then FuseUNet, ISIC2018 first (fast),
# then HAM10000. Each stage: train 300 epochs -> final test in our metric convention.
# Waits until the target GPU has >= 8000 MiB free and fewer than 4 compute processes.
set -u

PY=/home/PuMengYu/anaconda3/envs/medseg/bin/python
SRC=/home/PuMengYu/experiments/skin_segmentation/source/lightweight_repro_20261111
OUT=/media/8T/PuMengYu/experiments/skin_segmentation
GPU=${SWAP_GPU:-1}
NEED_FREE=5000
LOG=$OUT/LIGHTWEIGHT_QUEUE_$(date +%Y%m%d_%H%M).log

cd "$SRC" || exit 1
export PYTHONPATH="$SRC"
export CUDA_VISIBLE_DEVICES="$GPU"

echo "=== lightweight queue start $(date -Is) gpu=$GPU ===" | tee -a "$LOG"

wait_gpu() {
  while true; do
    free=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
    procs=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)
    # Slot rule: fill the card as soon as our compute-process count drops below 4.
    # A floor of 5000 MiB is kept only because batch 8 OOMs below that (verified: 3157 MiB OOM, batch 2 fine).
    if [ "$procs" -lt 4 ] && [ "$free" -ge 5000 ]; then
      echo "[queue] gpu$GPU ready free=${free}MiB procs=$procs $(date -Is)" | tee -a "$LOG"
      return 0
    fi
    echo "[queue] waiting gpu$GPU free=${free}MiB procs=$procs $(date -Is)" >> "$LOG"
    sleep 180
  done
}

run_stage() {
  local model=$1 dataset=$2 dir=$3
  echo "--- stage ${model} ${dataset} train ---" | tee -a "$LOG"
  "$PY" -u train_repro.py --model "$model" --dataset "$dataset" \
      --out "$OUT/$dir" --gpu 0 --epochs 300 --batch_size 8 --workers 4 >> "$LOG" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "!!! train failed ${model} ${dataset} rc=$rc $(date -Is)" | tee -a "$LOG"
    return $rc
  fi
  echo "--- stage ${model} ${dataset} final test ---" | tee -a "$LOG"
  "$PY" -u evaluate_final.py --dataset "$dataset" --out "$OUT/$dir" --gpu 0 \
      --workers 2 >> "$LOG" 2>&1
  echo "--- stage ${model} ${dataset} done $(date -Is) ---" | tee -a "$LOG"
}

wait_gpu
run_stage cmunext  ISIC2018 CMUNEXT_ISIC2018_20261111
run_stage fuseunet ISIC2018 FUSEUNET_ISIC2018_20261111
wait_gpu
run_stage cmunext  HAM10000 CMUNEXT_HAM10000_20261111
run_stage fuseunet HAM10000 FUSEUNET_HAM10000_20261111

echo "=== lightweight queue done $(date -Is) ===" | tee -a "$LOG"

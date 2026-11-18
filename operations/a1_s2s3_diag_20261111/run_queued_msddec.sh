#!/usr/bin/env bash
# Queued arm: A1+SwiftFormer-S with the MSDUNet decoder (scheme A).
# Waits until GPU1 has >= 3.5 GB free (used < 21000 MiB), then runs a shape check and starts
# the 300-epoch ISIC training. Never starts on top of a full card.
set -uo pipefail
PY=/home/PuMengYu/anaconda3/envs/medseg/bin/python
ROOT=/home/PuMengYu/experiments/skin_segmentation
OUT=/media/8T/PuMengYu/experiments/skin_segmentation
NAME=A1Swift_ISIC300_MSDdec_20261111
LOG=$OUT/${NAME}_queue.log
cd "$ROOT" || exit 1
export CUDA_VISIBLE_DEVICES=1

while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' ')
  if [ "$used" -lt 21000 ]; then
    echo "[queue] gpu1 used=${used}MiB -> shape check $(date -Is)" >> "$LOG"
    if "$PY" -B operations/a1_s2s3_diag_20261111/check_msddec.py >> "$LOG" 2>&1; then
      echo "[queue] shape check ok, starting training $(date -Is)" >> "$LOG"
      mkdir -p "$OUT/$NAME"
      "$PY" -B -u operations/a1_s2s3_diag_20261111/train_abl.py \
        --out "$OUT/$NAME" --dataset ISIC2018 --gpu 0 --epochs 300 \
        --batch-size 8 --workers 2 --t-max 300 --seed 42 \
        --candidate source.a1_swift_msddec_20261111.candidate --variant msddec \
        > "$OUT/$NAME/train.log" 2>&1
      echo "[queue] training exit=$? $(date -Is)" >> "$LOG"
    else
      echo "[queue] SHAPE CHECK FAILED, not starting $(date -Is)" >> "$LOG"
    fi
    break
  fi
  echo "[queue] waiting gpu1 used=${used}MiB $(date -Is)" >> "$LOG"
  sleep 120
done

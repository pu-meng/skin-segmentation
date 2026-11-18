#!/bin/bash
# Run the two remaining independent-training ablations serially on GPU1.
# A1 (no skips) and V1 (both skips) already completed with the identical protocol;
# V1-S2 and V1-S3 are new. Each starts from fresh pretrained weights, never from another
# group's checkpoint. 10 epochs each, same seed/data/optimiser/schedule.
set -u

PY=/home/PuMengYu/anaconda3/envs/medseg/bin/python
SRC=/home/PuMengYu/experiments/skin_segmentation
OUT=/media/8T/PuMengYu/experiments/skin_segmentation
TRAIN=$SRC/operations/a1_s2s3_diag_20261111/train_diag.py

cd "$SRC" || exit 1
export CUDA_VISIBLE_DEVICES=1

run() {
  local tag=$1 flag=$2 dir=$3
  echo "=== ${tag} start $(date -Is) ==="
  "$PY" -B -u "$TRAIN" --out "$OUT/$dir" --gpu 0 --epochs 10 $flag
  echo "=== ${tag} done rc=$? $(date -Is) ==="
}

run V1-S2 --no-skip2 A1_S2S3_V1S2_DIAG10_20261111
run V1-S3 --no-skip3 A1_S2S3_V1S3_DIAG10_20261111

# re-run the A1 control so its log also carries peak_gpu_mb
run A1 --no-skips A1_S2S3_DIAG10_A1_20261111

echo "=== all ablation runs finished $(date -Is) ==="

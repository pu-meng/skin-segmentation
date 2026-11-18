#!/usr/bin/env bash
# A1-S1-SKIPv0.6 (TinyViM-S) ablations on ISIC2018, 300 epochs each, one at a time (queued).
set -uo pipefail
PY=/home/PuMengYu/anaconda3/envs/medseg/bin/python
ROOT=/home/PuMengYu/experiments/skin_segmentation
OUT=/media/8T/PuMengYu/experiments/skin_segmentation
CAND=source.a1s1_skipv06_ablate_unet_20261111.candidate
cd "$ROOT" || exit 1
export CUDA_VISIBLE_DEVICES=1

run_one() {
  local name=$1 variant=$2
  mkdir -p "$OUT/$name"
  echo "[queue] $name start $(date -Is)" >> "$OUT/A1S1SKIPv06_ablation_queue.log"
  "$PY" -B -u operations/a1_s2s3_diag_20261111/train_abl.py \
    --out "$OUT/$name" --dataset ISIC2018 --gpu 0 --epochs 300 \
    --batch-size 8 --workers 2 --t-max 300 --seed 42 \
    --candidate "$CAND" --variant "$variant" \
    > "$OUT/$name/train.log" 2>&1
  echo "[queue] $name exit=$? $(date -Is)" >> "$OUT/A1S1SKIPv06_ablation_queue.log"
}

run_one A1S1SKIPv06_ISIC300_noSkip1_20261111 no_skip1
run_one A1S1SKIPv06_ISIC300_noE2_20261111 no_e2
run_one A1S1SKIPv06_ISIC300_noD2D3_20261111 no_d2d3
echo "[queue] all done $(date -Is)" >> "$OUT/A1S1SKIPv06_ablation_queue.log"

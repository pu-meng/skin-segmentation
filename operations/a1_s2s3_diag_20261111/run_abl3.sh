#!/usr/bin/env bash
# Three Skip2/Skip3 ablation arms on ISIC2018, 300 epochs each, all on physical GPU1.
#   arm1 keepS2 : Skip2 = full S2S3 block, Skip3 = A1 original      (--no-skip3)
#   arm2 keepS3 : Skip3 = full S2S3 block, Skip2 = A1 original      (--no-skip2)
#   arm3 convnextS : both skips = one encoder-side ConvNeXt block, no cross gate / no local DW
set -uo pipefail
PY=/home/PuMengYu/anaconda3/envs/medseg/bin/python
ROOT=/home/PuMengYu/experiments/skin_segmentation
OUT=/media/8T/PuMengYu/experiments/skin_segmentation
LOG=/home/PuMengYu/tmp_butest_20261111
mkdir -p "$LOG"
cd "$ROOT" || exit 1
export CUDA_VISIBLE_DEVICES=1

run_arm() {  # name  candidate  extra-flag
  local name=$1 cand=$2 extra=${3:-}
  local dir="$OUT/$name"
  mkdir -p "$dir"
  setsid nohup "$PY" -B -u operations/a1_s2s3_diag_20261111/train_abl.py \
    --out "$dir" --dataset ISIC2018 --gpu 0 --epochs 300 \
    --batch-size 8 --workers 2 --t-max 300 --seed 42 \
    --candidate "$cand" $extra \
    > "$dir/train.log" 2>&1 < /dev/null &
  echo "$name PID=$!"
}

run_arm A1S2S3_ISIC300_keepS2_20261111 source.a1_s2s3_20261111.candidate "--no-skip3"
sleep 5
run_arm A1S2S3_ISIC300_keepS3_20261111 source.a1_s2s3_20261111.candidate "--no-skip2"
sleep 5
run_arm A1S2S3_ISIC300_convnextS_20261111 source.a1_s2s3_convnext_20261111.candidate ""

#!/bin/bash
# AD-GBC (CVPR 2026) reproduction queue: ISIC2018 first, then HAM10000.
# Each stage: train (vendor protocol) -> final test in our metric convention.
set -u

PY=/home/PuMengYu/anaconda3/envs/medseg/bin/python
SRC=/home/PuMengYu/experiments/skin_segmentation/source/adgbc_repro_20261110
OUT=/media/8T/PuMengYu/experiments/skin_segmentation
GPU=0

cd "$SRC" || exit 1
export PYTHONPATH="$SRC"

echo "=== queue start $(date -Is) gpu=$GPU ==="

echo "--- stage 1: AD-GBC-L ISIC2018 train ---"
"$PY" train_adgbc.py --dataset ISIC2018 --arch GBC_Rolling_Unet_L \
    --out "$OUT/ADGBC_L_ISIC2018_20261110" --gpu $GPU
echo "--- stage 2: AD-GBC-L ISIC2018 final test (our convention) ---"
"$PY" evaluate_final.py --dataset ISIC2018 \
    --out "$OUT/ADGBC_L_ISIC2018_20261110" --gpu $GPU

echo "--- stage 3 SKIPPED (HAM10000 stage skipped at user request 2026-11-11) ---"
exit 0
echo "--- stage 3: AD-GBC-L HAM10000 train ---"
"$PY" train_adgbc.py --dataset HAM10000 --arch GBC_Rolling_Unet_L \
    --out "$OUT/ADGBC_L_HAM10000_20261110" --gpu $GPU
echo "--- stage 4: AD-GBC-L HAM10000 final test (our convention) ---"
"$PY" evaluate_final.py --dataset HAM10000 \
    --out "$OUT/ADGBC_L_HAM10000_20261110" --gpu $GPU

echo "=== queue done $(date -Is) ==="

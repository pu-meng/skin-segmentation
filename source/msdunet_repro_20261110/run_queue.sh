#!/bin/bash
set -e
cd /home/PuMengYu/experiments/skin_segmentation/source/msdunet_repro_20261110
PY=/home/PuMengYu/anaconda3/envs/medseg/bin/python
R=/media/8T/PuMengYu/experiments/skin_segmentation
echo "[queue] start $(date -Is) ISIC2018 on physical GPU1"
CUDA_VISIBLE_DEVICES=1 $PY -u train.py --dataset ISIC2018 --gpu 0 \
  --out $R/MSDUNET_ISIC2018_20261110 --epochs 200 --batch 8 --size 352 \
  > $R/MSDUNET_ISIC2018_20261110/train.log 2>&1
echo "[queue] ISIC2018 done $(date -Is), starting HAM10000"
CUDA_VISIBLE_DEVICES=1 $PY -u train.py --dataset HAM10000 --gpu 0 \
  --out $R/MSDUNET_HAM10000_20261110 --epochs 200 --batch 8 --size 352 \
  > $R/MSDUNET_HAM10000_20261110/train.log 2>&1
echo "[queue] HAM10000 done $(date -Is)"

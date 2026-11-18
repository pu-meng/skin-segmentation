#!/usr/bin/env bash
set -euo pipefail
cd /home/PuMengYu/experiments/skin_segmentation
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
OPS=/home/PuMengYu/experiments/skin_segmentation/operations/encoder_swap_A016_20261107
RUN=/media/8T/PuMengYu/experiments/skin_segmentation/EncoderSwap_A0_A1_A6_FRESH300_20261107
PY=/home/PuMengYu/anaconda3/envs/medseg/bin/python
if [ ! -f "$RUN/prepare_complete.json" ]; then
 "$PY" -B -u "$OPS/entry.py" prepare
fi
exec "$PY" -B -u "$OPS/entry.py" controller

#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd /home/PuMengYu/experiments/skin_segmentation
/home/PuMengYu/anaconda3/envs/medseg/bin/python -B -u /home/PuMengYu/experiments/skin_segmentation/operations/closeout_test6_20261107/entry.py controller 1 2>&1 | tee /home/PuMengYu/experiments/skin_segmentation/operations/closeout_test6_20261107/gpu1.log

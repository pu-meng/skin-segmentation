#!/bin/bash
set -euo pipefail
cd /home/PuMengYu/experiments/skin_segmentation
export PYTHONDONTWRITEBYTECODE=1 MKL_THREADING_LAYER=GNU
exec /home/PuMengYu/anaconda3/envs/medseg/bin/python -B -u /home/PuMengYu/experiments/skin_segmentation/operations/e13_slow_isolation_20261105/speed_lab.py

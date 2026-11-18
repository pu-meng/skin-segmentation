#!/usr/bin/env bash
set -euo pipefail
cd /home/PuMengYu/experiments/skin_segmentation
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
/home/PuMengYu/anaconda3/envs/medseg/bin/python -B -u /home/PuMengYu/experiments/skin_segmentation/operations/resume_tta_20261107/entry.py

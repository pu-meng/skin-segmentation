#!/usr/bin/env bash
set -euo pipefail
cd /home/PuMengYu/experiments/skin_segmentation
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
/home/PuMengYu/anaconda3/envs/medseg/bin/python -B -u operations/a1_skipc_v05_20261107/entry.py controller 2>&1 | tee -a /media/8T/PuMengYu/experiments/skin_segmentation/A1_SkipC_v05_FRESH300_20261107/controller.log

#!/usr/bin/env bash
set -euo pipefail
cd /home/PuMengYu/experiments/skin_segmentation
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
/home/PuMengYu/anaconda3/envs/medseg/bin/python -B -u operations/missing3_resume300_20261107/entry.py controller 2>&1 | tee -a operations/missing3_resume300_20261107/controller.log

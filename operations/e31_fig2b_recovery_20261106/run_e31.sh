#!/bin/bash
set -euo pipefail
cd /home/PuMengYu/experiments/skin_segmentation
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MKL_THREADING_LAYER=GNU
/home/PuMengYu/anaconda3/envs/medseg/bin/python -B -u /home/PuMengYu/experiments/skin_segmentation/operations/e31_fig2b_recovery_20261106/controller.py e31 2>&1 | tee -a /home/PuMengYu/experiments/skin_segmentation/operations/e31_fig2b_recovery_20261106/e31_controller_console.log

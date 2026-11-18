#!/bin/bash
set -euo pipefail
cd /home/PuMengYu/experiments/skin_segmentation/operations/engineering_speed_lab
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MKL_THREADING_LAYER=GNU
/home/PuMengYu/anaconda3/envs/medseg/bin/python -B -u speed_lab.py 2>&1 | tee -a console.log

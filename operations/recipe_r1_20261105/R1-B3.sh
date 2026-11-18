#!/usr/bin/env bash
set -euo pipefail
cd /home/PuMengYu/experiments/skin_segmentation
export R1_ARM=R1-B3
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
/home/PuMengYu/anaconda3/envs/medseg/bin/python -B -u /home/PuMengYu/experiments/skin_segmentation/operations/recipe_r1_20261105/arm_controller.py 2>&1 | tee /home/PuMengYu/experiments/skin_segmentation/log/TinyViM_R1统一配方_HAM_20261105/R1-B3/controller_console.log

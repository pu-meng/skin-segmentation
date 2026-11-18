#!/bin/bash
set -euo pipefail
cd /home/PuMengYu/experiments/skin_segmentation
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MKL_THREADING_LAYER=GNU
/home/PuMengYu/anaconda3/envs/medseg/bin/python -B -u -m source.wpo_mr_20261105.arm_controller 2>&1 | tee '/home/PuMengYu/experiments/skin_segmentation/log/TinyViM_B-WPO-MR_HAM_20261105/B-WPO-MR/tmux_console.log'

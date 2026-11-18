#!/bin/bash
set -euo pipefail
cd /home/PuMengYu/experiments/skin_segmentation
export WPO_ARM=WPO-S1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MKL_THREADING_LAYER=GNU
/home/PuMengYu/anaconda3/envs/medseg/bin/python -B -u -m source.wpo_skips_20261105.arm_controller 2>&1 | tee 'log/TinyViM_WPO单Skip位置_HAM_20261105/WPO-S1/tmux_console.log'

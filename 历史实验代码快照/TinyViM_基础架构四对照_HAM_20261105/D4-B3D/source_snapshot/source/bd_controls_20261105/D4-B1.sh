#!/bin/bash
set -euo pipefail
cd /home/PuMengYu/experiments/skin_segmentation
export BD_ARM=D4-B1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MKL_THREADING_LAYER=GNU
mkdir -p '/home/PuMengYu/experiments/skin_segmentation/log/TinyViM_基础架构四对照_HAM_20261105/D4-B1'
/home/PuMengYu/anaconda3/envs/medseg/bin/python -B -u -m source.bd_controls_20261105.arm_controller 2>&1 | tee '/home/PuMengYu/experiments/skin_segmentation/log/TinyViM_基础架构四对照_HAM_20261105/D4-B1/tmux_console.log'

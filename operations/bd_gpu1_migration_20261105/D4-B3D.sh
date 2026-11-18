#!/bin/bash
set -euo pipefail
cd /home/PuMengYu/experiments/skin_segmentation
export BD_ARM=D4-B3D PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MKL_THREADING_LAYER=GNU
/home/PuMengYu/anaconda3/envs/medseg/bin/python -B -u /home/PuMengYu/experiments/skin_segmentation/operations/bd_gpu1_migration_20261105/arm_controller_gpu1.py 2>&1 | tee '/home/PuMengYu/experiments/skin_segmentation/log/TinyViM_基础架构四对照_HAM_20261105/D4-B3D/tmux_gpu1_console.log'

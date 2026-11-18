#!/bin/bash
set -euo pipefail
cd /home/PuMengYu/experiments/skin_segmentation
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
/home/PuMengYu/anaconda3/envs/medseg/bin/python -B -u /home/PuMengYu/experiments/skin_segmentation/source/e123_joint_20261105/controller.py 2>&1 | tee /home/PuMengYu/experiments/skin_segmentation/log/TinyViM_E123.1_联合新版_HAM_20261105/tmux_console.log

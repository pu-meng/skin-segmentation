#!/bin/bash
set -euo pipefail
cd /home/PuMengYu/experiments/skin_segmentation
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
/home/PuMengYu/anaconda3/envs/medseg/bin/python -u operations/e1232_fourarm_screen_20261106/controller.py 2>&1 | tee log/TinyViM_E1232_FourArm_SCREEN100of300_20261106/controller_console.log

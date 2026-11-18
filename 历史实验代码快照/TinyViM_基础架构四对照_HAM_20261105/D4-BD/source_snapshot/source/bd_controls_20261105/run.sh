#!/bin/bash
set -euo pipefail
cd /home/PuMengYu/experiments/skin_segmentation
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
/home/PuMengYu/anaconda3/envs/medseg/bin/python -B -u -m source.bd_controls_20261105.wait_queue 2>&1 | tee '/home/PuMengYu/experiments/skin_segmentation/log/TinyViM_基础架构四对照_HAM_20261105/queue_console.log'

#!/bin/bash
set -euo pipefail
cd /home/PuMengYu/experiments/skin_segmentation
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
/home/PuMengYu/anaconda3/envs/medseg/bin/python -B -u /home/PuMengYu/experiments/skin_segmentation/operations/wpo_mr_20261105/wait_queue.py 2>&1 | tee '/home/PuMengYu/experiments/skin_segmentation/log/TinyViM_B-WPO-MR_HAM_20261105/queue_console.log'

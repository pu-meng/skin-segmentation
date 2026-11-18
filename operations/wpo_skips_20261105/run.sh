#!/bin/bash
set -euo pipefail
cd /home/PuMengYu/experiments/skin_segmentation
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
/home/PuMengYu/anaconda3/envs/medseg/bin/python -B -u operations/wpo_skips_20261105/wait_queue.py 2>&1 | tee log/TinyViM_WPO单Skip位置_HAM_20261105/queue_console.log

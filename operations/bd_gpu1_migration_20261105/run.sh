#!/bin/bash
set -euo pipefail
cd /home/PuMengYu/experiments/skin_segmentation
/home/PuMengYu/anaconda3/envs/medseg/bin/python -B -u /home/PuMengYu/experiments/skin_segmentation/operations/bd_gpu1_migration_20261105/wait_gpu1_pair.py 2>&1 | tee /home/PuMengYu/experiments/skin_segmentation/operations/bd_gpu1_migration_20261105/queue_console.log

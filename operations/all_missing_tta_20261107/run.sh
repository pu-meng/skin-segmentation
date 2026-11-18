#!/bin/bash
set -o pipefail
cd /home/PuMengYu/experiments/skin_segmentation || exit 1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
/home/PuMengYu/anaconda3/envs/medseg/bin/python -B -u /home/PuMengYu/experiments/skin_segmentation/operations/all_missing_tta_20261107/controller.py 2>&1 | tee /home/PuMengYu/experiments/skin_segmentation/operations/all_missing_tta_20261107/controller_console.log

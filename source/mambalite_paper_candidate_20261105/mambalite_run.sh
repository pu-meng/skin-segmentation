#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/PuMengYu/experiments/skin_segmentation
SRC=$ROOT/source/mambalite_paper_candidate_20261105
LOG=$ROOT/log/mambalite_fig2b_20261105
mkdir -p "$LOG"
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
/home/PuMengYu/anaconda3/envs/medseg/bin/python -u "$SRC/controller.py" 2>&1 | tee -a "$LOG/controller_console.log"

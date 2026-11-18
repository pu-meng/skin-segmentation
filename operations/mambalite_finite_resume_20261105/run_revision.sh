#!/bin/bash
set -uo pipefail
cd /home/PuMengYu/experiments/skin_segmentation
O=/home/PuMengYu/experiments/skin_segmentation/operations/mambalite_finite_resume_20261105
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
/home/PuMengYu/anaconda3/envs/medseg/bin/python -u "$O/controller_revision.py" 2>&1 | tee "$O/controller_revision.log"
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > "$O/controller_revision.exitcode"
exit "$rc"

#!/usr/bin/env bash
set -euo pipefail
case "${1:-}" in Full|NoE2) ;; *) exit 2;; esac
cd /home/PuMengYu/experiments/skin_segmentation
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
/home/PuMengYu/anaconda3/envs/medseg/bin/python -B -u /home/PuMengYu/experiments/skin_segmentation/operations/candidateA_E2_pair100_20261107/entry.py controller "$1"

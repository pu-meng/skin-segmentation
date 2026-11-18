#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT=/home/PuMengYu/experiments/skin_segmentation
PYTHON=/home/PuMengYu/experiments/MambaLiteUNet/.venv/bin/python
GPU=${1:-1}
[[ "$GPU" =~ ^[0-9]+$ ]] || exit 64
cd "$PROJECT"
test -x "$PYTHON"
test -f "$PROJECT/log/ISIC2018_瓶颈对比_20261031/parallel_adoption.json"
export CUDA_VISIBLE_DEVICES="$GPU" EXPERIMENT_PHYSICAL_GPU="$GPU"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
exec "$PYTHON" -u source/isic2018/parallel_controller.py

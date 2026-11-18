#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT=/home/PuMengYu/experiments/skin_segmentation
PYTHON=/home/PuMengYu/experiments/MambaLiteUNet/.venv/bin/python
GPU=${1:-1}
[[ "$GPU" =~ ^[0-9]+$ ]] || exit 64
cd "$PROJECT"
test -x "$PYTHON"
test -f "$PROJECT/data/processed/isic2018_256_dedup_seed42_20261031/cache_complete.json"
FREE=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits)
[[ "$FREE" =~ ^[0-9]+$ ]] || exit 75
if (( FREE < 8000 )); then
 printf 'GPU%s free=%s MiB; need8000 MiB\n' "$GPU" "$FREE" >&2
 exit 75
fi
export CUDA_VISIBLE_DEVICES="$GPU" EXPERIMENT_PHYSICAL_GPU="$GPU"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
exec "$PYTHON" -u source/isic2018/controller.py

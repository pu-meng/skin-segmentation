#!/usr/bin/env bash
set -Eeuo pipefail
P=/home/PuMengYu/experiments/skin_segmentation
ROOT="$P/log/UNet2D_版本2_CUDA等价试跑3轮_20261029"
PY=/home/PuMengYu/experiments/MambaLiteUNet/.venv/bin/python
cd "$P"
mkdir "$ROOT"
exec 9>"$ROOT/controller.lock"
flock -n 9 || exit 73
trap 'code=$?; printf "%s\n" "$code" > "$ROOT/controller.exitcode"' EXIT
export CUDA_VISIBLE_DEVICES=1
export EXPERIMENT_PHYSICAL_GPU=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export CUDA_PILOT_TEST_REPORT="$ROOT/equivalence_test.json"
"$PY" -u source/bottleneck_mamba/test_v2_cuda_pilot.py 2>&1 | tee "$ROOT/equivalence_console.log"
"$PY" -u source/bottleneck_mamba/experiment_v2_cuda_pilot.py --root "$ROOT" 2>&1 | tee "$ROOT/console.log"

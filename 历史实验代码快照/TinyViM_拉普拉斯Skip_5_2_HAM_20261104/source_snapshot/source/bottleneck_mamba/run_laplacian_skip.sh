#!/usr/bin/env bash
set -euo pipefail
project=/home/PuMengYu/experiments/skin_segmentation
python_bin=/home/PuMengYu/anaconda3/envs/medseg/bin/python
run_root="$project/log/TinyViM_拉普拉斯Skip_5_2_HAM_20261104"
physical_gpu="${1:-0}"
[[ -x "$python_bin" ]]
[[ "$physical_gpu" =~ ^[0-9]+$ ]]
mkdir -p "$run_root"
exec 9>"$run_root/controller.lock"
flock -n 9
[[ ! -e "$run_root/run/status.json" ]]
[[ ! -e "$run_root/run/checkpoints" ]]
free_memory=$(nvidia-smi -i "$physical_gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
(( free_memory >= 12000 ))
export CUDA_VISIBLE_DEVICES="$physical_gpu"
export EXPERIMENT_PHYSICAL_GPU="$physical_gpu"
export PYTHONUNBUFFERED=1
cd "$project"
"$python_bin" -u source/bottleneck_mamba/experiment_laplacian_skip.py --mode preflight --run-root "$run_root" --gpu 0 > "$run_root/preflight_console.log" 2>&1
"$python_bin" -u source/bottleneck_mamba/experiment_laplacian_skip.py --mode train --run-root "$run_root" --gpu 0 > "$run_root/train_console.log" 2>&1
"$python_bin" -u source/bottleneck_mamba/final_test_laplacian_skip.py --arm laplacian --gpu 0 > "$run_root/test_console.log" 2>&1

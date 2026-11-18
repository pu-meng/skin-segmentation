#!/usr/bin/env bash
set -euo pipefail
project=/home/PuMengYu/experiments/skin_segmentation
python_bin=/home/PuMengYu/anaconda3/envs/medseg/bin/python
run_root="$project/log/TinyViM_四层对应Decoder_基础_HAM_20261104"
physical_gpu="${1:-0}"
min_free_gib="${TRAIN_MIN_FREE_GIB:-12}"
[[ -x "$python_bin" ]]
[[ "$physical_gpu" =~ ^[0-9]+$ && "$min_free_gib" =~ ^[0-9]+$ ]]
mkdir -p "$run_root"
exec 9>"$run_root/controller.lock"
flock -n 9
[[ ! -e "$run_root/run/status.json" && ! -e "$run_root/run/checkpoints" ]]
free_mib=$(nvidia-smi -i "$physical_gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
if (( free_mib < min_free_gib * 1024 )); then
    printf '训练需要至少 %s GiB 空闲显存，当前 %s MiB\n' "$min_free_gib" "$free_mib" >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES="$physical_gpu"
export EXPERIMENT_PHYSICAL_GPU="$physical_gpu"
export PYTHONUNBUFFERED=1
cd "$project"
"$python_bin" -u source/bottleneck_mamba/experiment_four_stage_decoder.py --mode preflight --gpu 0 > "$run_root/preflight_console.log" 2>&1
"$python_bin" -u source/bottleneck_mamba/experiment_four_stage_decoder.py --mode train --gpu 0 > "$run_root/train_console.log" 2>&1
"$python_bin" -u source/bottleneck_mamba/final_test_four_stage_decoder.py --arm four_stage --gpu 0 > "$run_root/test_console.log" 2>&1

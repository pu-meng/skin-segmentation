#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT=/home/PuMengYu/experiments/skin_segmentation
PYTHON=/home/PuMengYu/experiments/MambaLiteUNet/.venv/bin/python
RUN="$PROJECT/log/UNet2D_瓶颈Mamba_版本2"
cd "$PROJECT"
mkdir -p "$RUN"
exec 9>"$RUN/tmux_owner.lock"
flock -n 9 || exit 73
trap 'code=$?; printf "%s\n" "$code" > "$RUN/controller_exitcode"; printf "UNet V2 exited: %s\n" "$code"' EXIT
export CUDA_VISIBLE_DEVICES=1
export EXPERIMENT_PHYSICAL_GPU=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
printf '%s\n' "$$" > "$RUN/controller.pid"
printf 'UNet2D+V2; authorized shared physical GPU1; utilization is not a wait condition.\n'
nvidia-smi -i 1 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
"$PYTHON" -u source/bottleneck_mamba/experiment_unet_v2.py --arm unet_v2 --mode preflight --run-root "$RUN" --gpu 0 2>&1 | tee "$RUN/preflight_console.log"
printf '\n=== FRESH FORMAL TRAINING ===\n'
"$PYTHON" -u source/bottleneck_mamba/experiment_unet_v2.py --arm unet_v2 --mode train --run-root "$RUN" --gpu 0 2>&1 | tee "$RUN/training_console.log"

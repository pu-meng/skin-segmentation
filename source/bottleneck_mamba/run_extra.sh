#!/usr/bin/env bash
set -Eeuo pipefail
ARM=${1:?tinyvim_v2 or unet_v1}
GPU=${2:?physical GPU index}
PROJECT=/home/PuMengYu/experiments/skin_segmentation
PYTHON=/home/PuMengYu/experiments/MambaLiteUNet/.venv/bin/python
case "$ARM" in
 tinyvim_v2) RUN="$PROJECT/log/瓶颈Mamba_版本2";;
 unet_v1) RUN="$PROJECT/log/UNet2D_瓶颈Mamba_版本1";;
 *) printf 'Unknown arm\n';exit 64;;
esac
cd "$PROJECT"
mkdir -p "$RUN"
exec 9>"$RUN/tmux_owner.lock"
flock -n 9 || { printf 'Another controller owns this arm\n';exit 73; }
trap 'code=$?; printf "%s\n" "$code" > "$RUN/controller_exitcode"; printf "Controller exited: %s\n" "$code"' EXIT
printf 'Authorized shared GPU%s; utilization is not an idle/wait condition.\n' "$GPU"
nvidia-smi -i "$GPU" --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
export CUDA_VISIBLE_DEVICES="$GPU"
export EXPERIMENT_PHYSICAL_GPU="$GPU"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
printf '%s\n' "$$" > "$RUN/controller.pid"
printf 'ARM=%s GPU=%s; 300epoch batch8, no V1+V2 combination, no test loader.\n' "$ARM" "$GPU"
printf '\n=== REAL-DATA PREFLIGHT ===\n'
"$PYTHON" -u source/bottleneck_mamba/experiment_extra.py --arm "$ARM" --mode preflight --run-root "$RUN" --gpu 0 2>&1 | tee "$RUN/preflight_console.log"
printf '\n=== FRESH FORMAL TRAINING ===\n'
"$PYTHON" -u source/bottleneck_mamba/experiment_extra.py --arm "$ARM" --mode train --run-root "$RUN" --gpu 0 2>&1 | tee "$RUN/training_console.log"
printf '\n=== TRAINING AND VALIDATION PROCESS RETURNED ===\n'

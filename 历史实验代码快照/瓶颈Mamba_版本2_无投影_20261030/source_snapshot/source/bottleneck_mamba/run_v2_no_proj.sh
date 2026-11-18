#!/usr/bin/env bash
# Each arm: GPU preflight -> fresh 300 epochs -> validation-best independent test.
set -Eeuo pipefail
ARM=${1:?tinyvim_v2_no_proj or unet_v2_no_proj}
GPU=${2:-1}
PROJECT=/home/PuMengYu/experiments/skin_segmentation
PYTHON=/home/PuMengYu/experiments/MambaLiteUNet/.venv/bin/python
case "$ARM" in
 tinyvim_v2_no_proj) RUN="$PROJECT/log/瓶颈Mamba_版本2_无投影_20261030"; MIN_FREE=6000;;
 unet_v2_no_proj) RUN="$PROJECT/log/UNet2D_瓶颈Mamba_版本2_无投影_20261030"; MIN_FREE=8000;;
 *) printf 'Unknown arm: %s\n' "$ARM"; exit 64;;
esac
[[ "$GPU" =~ ^[0-9]+$ ]] || exit 64
cd "$PROJECT"
test -x "$PYTHON"
mkdir -p "$RUN"
exec 9>"$RUN/tmux_owner.lock"
flock -n 9 || exit 73
trap 'code=$?; printf "%s\n" "$code" > "$RUN/controller_exitcode"' EXIT
if [[ -e "$RUN/run/status.json" || -e "$RUN/run/checkpoints" ]]; then
 printf 'Existing training; refusing overwrite: %s\n' "$RUN"; exit 74
fi
if FREE=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits 2>&1); then
 :
else
 printf 'Cannot access GPU%s; training not started: %s\n' "$GPU" "$FREE" >&2
 exit 75
fi
[[ "$FREE" =~ ^[0-9]+$ ]] || exit 75
if (( FREE < MIN_FREE )); then
 printf 'GPU%s free=%s MiB; need=%s MiB\n' "$GPU" "$FREE" "$MIN_FREE"; exit 75
fi
export CUDA_VISIBLE_DEVICES="$GPU" EXPERIMENT_PHYSICAL_GPU="$GPU"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
printf '%s\n' "$$" > "$RUN/controller.pid"
printf 'ARM=%s GPU=%s: preflight -> train -> final test\n' "$ARM" "$GPU"
"$PYTHON" -u source/bottleneck_mamba/experiment_v2_no_proj.py --arm "$ARM" --mode preflight --run-root "$RUN" --gpu 0 2>&1 | tee "$RUN/preflight_console.log"
"$PYTHON" -u source/bottleneck_mamba/experiment_v2_no_proj.py --arm "$ARM" --mode train --run-root "$RUN" --gpu 0 2>&1 | tee "$RUN/training_console.log"
"$PYTHON" -u source/bottleneck_mamba/final_test_pipeline.py --arm "$ARM" --output-root "$PROJECT/log/瓶颈Mamba_版本2_无投影_最终测试_20261030" --gpu 0 2>&1 | tee "$RUN/final_test_console.log"

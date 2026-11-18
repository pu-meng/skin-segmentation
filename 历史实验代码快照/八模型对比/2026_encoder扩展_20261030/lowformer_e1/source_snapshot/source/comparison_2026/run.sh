#!/usr/bin/env bash
# Approved encoder extension: real preflight -> fresh training -> fixed-best test.
set -Eeuo pipefail
MODEL=${1:?lowformer_e1 or cpubone_b0_bftraining}
GPU=${2:-0}
PROJECT=/home/PuMengYu/experiments/skin_segmentation
PYTHON=/home/PuMengYu/experiments/MambaLiteUNet/.venv/bin/python
case "$MODEL" in
 lowformer_e1|cpubone_b0_bftraining) ;;
 *) printf 'Unapproved model: %s\n' "$MODEL"; exit 64;;
esac
[[ "$GPU" =~ ^[0-9]+$ ]] || exit 64
RUN="$PROJECT/log/八模型对比/2026_encoder扩展_20261030/$MODEL"
cd "$PROJECT"
test -x "$PYTHON"
mkdir -p "$RUN"
exec 9>"$RUN/controller.lock"
flock -n 9 || exit 73
trap 'code=$?; printf "%s\n" "$code" > "$RUN/controller_exitcode"' EXIT
if [[ -e "$RUN/run/status.json" || -e "$RUN/run/checkpoints" ]]; then
 printf 'Existing run; refusing overwrite: %s\n' "$RUN"; exit 74
fi
FREE=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits)
[[ "$FREE" =~ ^[0-9]+$ ]] || exit 75
if (( FREE < 6000 )); then
 printf 'GPU%s free=%s MiB; need 6000 MiB\n' "$GPU" "$FREE"; exit 75
fi
export CUDA_VISIBLE_DEVICES="$GPU" EXPERIMENT_PHYSICAL_GPU="$GPU"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
printf '%s\n' "$$" > "$RUN/controller.pid"
"$PYTHON" -u source/comparison_2026/experiment.py --arm "$MODEL" --mode preflight --run-root "$RUN" --gpu 0 2>&1 | tee "$RUN/preflight_console.log"
"$PYTHON" -u source/comparison_2026/experiment.py --arm "$MODEL" --mode train --run-root "$RUN" --gpu 0 2>&1 | tee "$RUN/training_console.log"
"$PYTHON" -u source/comparison_2026/final_test.py --arm "$MODEL" --gpu 0 2>&1 | tee "$RUN/final_test_console.log"
"$PYTHON" -u source/comparison_2026/summarize.py

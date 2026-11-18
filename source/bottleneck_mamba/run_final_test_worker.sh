#!/usr/bin/env bash
set -Eeuo pipefail
WORKER=${1:?worker name required}
PROJECT=/home/PuMengYu/experiments/skin_segmentation
ROOT="$PROJECT/log/瓶颈Mamba_最终测试_20261029"
PYTHON=/home/PuMengYu/experiments/MambaLiteUNet/.venv/bin/python
cd "$PROJECT"
mkdir -p "$ROOT"
exec 9>"$ROOT/controller_${WORKER}.lock"
flock -n 9 || exit 73
trap 'code=$?; printf "%s\n" "$code" > "$ROOT/controller_'"$WORKER"'.exitcode"; printf "TEST worker exited: %s\n" "$code"' EXIT
export CUDA_VISIBLE_DEVICES=1
export EXPERIMENT_PHYSICAL_GPU=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
run_test() {
 local arm="$1";shift
 "$PYTHON" -u source/bottleneck_mamba/final_test_pipeline.py --arm "$arm" --output-root "$ROOT" --gpu 0 "$@" 2>&1 | tee "$ROOT/${arm}_console.log"
}
case "$WORKER" in
 tinyvim_v1) run_test tinyvim_v1;run_test tinyvim_base;;
 tinyvim_v2) run_test tinyvim_v2;;
 unet_v1) run_test unet_v1;run_test unet_base;;
 unet_v2_auto) run_test unet_v2 --wait-for-training;;
 *) printf 'Unknown worker\n';exit 64;;
esac

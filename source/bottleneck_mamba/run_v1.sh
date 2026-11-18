#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT=/home/PuMengYu/experiments/skin_segmentation
PYTHON=/home/PuMengYu/experiments/MambaLiteUNet/.venv/bin/python
RUN="$PROJECT/log/瓶颈Mamba_版本1"
cd "$PROJECT"
mkdir -p "$RUN"
exec 9>"$RUN/tmux_owner.lock"
flock -n 9 || { printf 'Another V1 controller owns this run.\n'; exit 73; }
trap 'code=$?; printf "%s\n" "$code" > "$RUN/controller_exitcode"; printf "V1 controller exited: %s\n" "$code"' EXIT
# This launch has no permission to share another task on GPU0.
occupied=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits)
if [[ -n "$occupied" ]]; then
    printf 'GPU0 acquired another task; refusing to share: %s\n' "$occupied"
    exit 75
fi
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
printf 'V1 experiment. Bottleneck only; no V2 combination. GPU0.\n'
printf '300 epochs, batch8, same HAM10000 split; validation only, test deferred.\n'
printf '%s\n' "$$" > "$RUN/controller.pid"
printf '\n=== REAL-DATA PREFLIGHT ===\n'
"$PYTHON" -u source/bottleneck_mamba/experiment_v1.py --mode preflight --run-root "$RUN" --gpu 0 2>&1 | tee "$RUN/preflight_console.log"
printf '\n=== FRESH FORMAL TRAINING (preflight weights discarded) ===\n'
"$PYTHON" -u source/bottleneck_mamba/experiment_v1.py --mode train --run-root "$RUN" --gpu 0 2>&1 | tee "$RUN/training_console.log"
printf '\n=== V1 TRAINING AND VALIDATION PROCESS RETURNED ===\n'

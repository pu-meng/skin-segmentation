#!/usr/bin/env bash
# HAM fixed protocol: candidate contracts -> GPU preflight -> 300 epochs -> complete test.
set -Eeuo pipefail
ARM=${1:?tinyvim_v2_shared_a_mean or unet_v2_shared_a_mean}
GPU=${2:-1}
PROJECT=/home/PuMengYu/experiments/skin_segmentation
PYTHON=/home/PuMengYu/experiments/MambaLiteUNet/.venv/bin/python
case "$ARM" in
 tinyvim_v2_shared_a_mean) RUN="$PROJECT/log/瓶颈Mamba_V2共享A平均读出_TinyViM_20261031"; MIN_FREE_GIB=${TRAIN_MIN_FREE_GIB:-6};;
 unet_v2_shared_a_mean) RUN="$PROJECT/log/瓶颈Mamba_V2共享A平均读出_UNet2D_20261031"; MIN_FREE_GIB=${TRAIN_MIN_FREE_GIB:-8};;
 *) printf 'Unknown arm: %s\n' "$ARM" >&2; exit 64;;
esac
[[ "$GPU" =~ ^[0-9]+$ && "$MIN_FREE_GIB" =~ ^[0-9]+$ ]] || exit 64
(( MIN_FREE_GIB > 0 )) || exit 64
cd "$PROJECT"
test -x "$PYTHON"
mkdir -p "$RUN"
exec 9>"$RUN/controller.lock"
flock -n 9 || exit 73
trap 'code=$?; printf "%s\n" "$code" > "$RUN/controller_exitcode"' EXIT
if [[ -e "$RUN/run/status.json" || -e "$RUN/run/checkpoints" ]]; then
 printf 'Existing training; refusing overwrite: %s\n' "$RUN" >&2; exit 74
fi
export CUDA_VISIBLE_DEVICES="$GPU" EXPERIMENT_PHYSICAL_GPU="$GPU"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
printf '%s\n' "$$" > "$RUN/controller.pid"
"$PYTHON" -u source/bottleneck_mamba/experiment_v2_shared_a_mean.py --arm "$ARM" --mode prepare --run-root "$RUN" --gpu 0 2>&1 | tee "$RUN/prepare_console.log"
"$PYTHON" - "$RUN" "$GPU" "$MIN_FREE_GIB" <<'PY'
import json,subprocess,sys
from datetime import datetime
from pathlib import Path
root=Path(sys.argv[1]);gpu=sys.argv[2];minimum=int(sys.argv[3])*1024
try:
    result=subprocess.run(['nvidia-smi','-i',gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    free=int(result.stdout.strip())
    if free<minimum:raise RuntimeError(f'GPU {gpu}: free {free} MiB < required {minimum} MiB')
except Exception as error:
    detail=getattr(error,'stderr','') or repr(error)
    record={'status':'not_started','stage':'gpu_resource_check','physical_gpu':int(gpu),
            'required_free_mib':minimum,'error':detail,'time':datetime.now().astimezone().isoformat()}
    (root/'launch_status.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(record,ensure_ascii=False),flush=True)
    sys.exit(75)
PY
"$PYTHON" -u source/bottleneck_mamba/test_v2_shared_a_mean.py --device cuda --report "$RUN/gpu_contract.json" 2>&1 | tee "$RUN/gpu_contract_console.log"
"$PYTHON" -u source/bottleneck_mamba/experiment_v2_shared_a_mean.py --arm "$ARM" --mode preflight --run-root "$RUN" --gpu 0 2>&1 | tee "$RUN/preflight_console.log"
"$PYTHON" -u source/bottleneck_mamba/experiment_v2_shared_a_mean.py --arm "$ARM" --mode train --run-root "$RUN" --gpu 0 2>&1 | tee "$RUN/training_console.log"
"$PYTHON" -u source/bottleneck_mamba/final_test_shared_a_mean.py --arm "$ARM" --gpu 0 --min-free-gib "${EVAL_MIN_FREE_GIB:-2}" 2>&1 | tee "$RUN/final_test_console.log"

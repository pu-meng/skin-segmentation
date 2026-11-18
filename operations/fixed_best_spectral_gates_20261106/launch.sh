#!/usr/bin/env bash
set -uo pipefail
cd /home/PuMengYu/experiments/skin_segmentation
OUT=/media/8T/PuMengYu/diagnostics/fixed_best_spectral_gates_20261106
OPS=/home/PuMengYu/experiments/skin_segmentation/operations/fixed_best_spectral_gates_20261106
exec 9>"$OUT/controller.lock"
flock -n 9 || exit 73
printf '%s\n' "$$" > "$OUT/controller.pid"
export CUDA_VISIBLE_DEVICES=1 BD_ARM=D4-B3D PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
start=$SECONDS
while true; do
 free=$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits)
 if [ "${free:-0}" -ge 4500 ]; then break; fi
 printf '{"status":"waiting_resources","physical_gpu":1,"free_mib":%s,"controller_pid":%s}\n' "${free:-0}" "$$" > "$OUT/status.json"
 if [ $((SECONDS-start)) -gt 86400 ]; then printf '124\n' > "$OUT/exitcode"; exit 124; fi
 sleep 180
done
/home/PuMengYu/anaconda3/envs/medseg/bin/python -B -u "$OPS/run.py" run 2>&1 | tee "$OUT/stdout.log"
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > "$OUT/exitcode"
exit "$rc"

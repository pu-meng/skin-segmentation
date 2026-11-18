#!/bin/bash
set -Eeuo pipefail
OP=/home/PuMengYu/experiments/skin_segmentation/operations/duala_best39_test_20261107
OUT=/media/8T/PuMengYu/experiments/skin_segmentation/DecoderDualA_SCREEN100of300_20261106/screen100_best39_test_20261107
mkdir -p "$OUT"
exec 9>"$OP/launch.lock"
flock -n 9 || exit 73
echo $$ > "$OP/controller.pid"
export CUDA_VISIBLE_DEVICES=GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b
export EXPERIMENT_PHYSICAL_GPU=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd /home/PuMengYu/experiments/skin_segmentation
set +e
timeout --foreground --signal=TERM 3600 /home/PuMengYu/anaconda3/envs/medseg/bin/python -B -u "$OP/test.py" 2>&1 | tee "$OUT/stdout.log"
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > "$OP/exitcode"
exit "$rc"

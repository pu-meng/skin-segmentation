#!/usr/bin/env bash
set -euo pipefail
cd /home/PuMengYu/experiments/skin_segmentation
D=/home/PuMengYu/experiments/skin_segmentation/source/three_module_impl_20261104
R=/home/PuMengYu/experiments/skin_segmentation/log/TinyViM_三模块_E1_DWPO_HAM_20261104
exec 9>"$R/controller.lock"
flock -n 9 || exit 90
export CUDA_VISIBLE_DEVICES=1 EXPERIMENT_PHYSICAL_GPU=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
PY=/home/PuMengYu/anaconda3/envs/medseg/bin/python
echo "CONTROLLER $$ $(date --iso-8601=seconds)"
echo $$ > "$R/controller.pid"
trap 'rc=$?; printf "%s\n" "$rc" > "$R/controller.exitcode"' EXIT
for phase in preflight train test; do
 # No sharing, and no stealing idle gaps: all previously active controllers are GPU0 only.
 used=$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader,nounits)
 if [ -n "$used" ]; then echo "BLOCKED GPU1 has compute PID: $used";exit 91;fi
 echo "PHASE $phase $(date --iso-8601=seconds)"
 "$PY" -u "$D/entry.py" "$phase" 2>&1 | tee "$R/${phase}_console.log"
done

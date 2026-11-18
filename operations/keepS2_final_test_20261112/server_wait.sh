#!/usr/bin/env bash
set -eu
cd /home/PuMengYu/experiments/skin_segmentation
BASE=/media/8T/PuMengYu/experiments/skin_segmentation/EXPLORATION_REPAIR_FRESH300_20261112
GPU=GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b
exec 9>"$BASE/controller.lock"
flock -n 9 || { printf '%s\n' 'BLOCKED: another controller holds lock'; exit 2; }
if [ -e "$BASE/keepS2/final_test" ]; then printf '%s\n' 'BLOCKED: final_test already exists'; exit 2; fi
printf 'SERVER_WAITING pid=%s started=%s\n' "$$" "$(date -Is)"
while :; do
 free=$(nvidia-smi --query-gpu=uuid,memory.free --format=csv,noheader,nounits | python3 -c 'import sys; print(next(x.split(",")[1].strip() for x in sys.stdin if x.split(",")[0].strip()=="GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b"))')
 printf 'CHECK time=%s GPU1_FREE_MIB=%s threshold=6000\n' "$(date -Is)" "$free"
 if [ "$free" -ge 6000 ]; then break; fi
 sleep 60
done
export CUDA_VISIBLE_DEVICES="$GPU" EXPERIMENT_PHYSICAL_GPU=1 EXPERIMENT_GPU_UUID="$GPU" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
printf 'TEST_START time=%s\n' "$(date -Is)"
/home/PuMengYu/anaconda3/envs/medseg/bin/python -B -u -c 'import runpy; m=runpy.run_path("operations/exploration_protocol_repair_20261112/runner.py",run_name="keepS2_test_only"); m["phase"]("keepS2","ISIC2018","test")'
printf 'TEST_FINISHED time=%s\n' "$(date -Is)"

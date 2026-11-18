#!/usr/bin/env bash
# Real preflight -> independent full training -> fixed-best independent test.
set -Eeuo pipefail
CANDIDATE=${1:?candidate name required}
PHYSICAL_GPU=${2:?physical GPU required}
PROJECT=/home/PuMengYu/experiments/skin_segmentation
PYTHON=/home/PuMengYu/anaconda3/envs/medseg/bin/python
case "$CANDIDATE" in
  convneur_m2|microvitv2_s1|locatvit_tiny|visionhope_tiny) ;;
  *) printf 'Unsupported candidate: %s\n' "$CANDIDATE"; exit 64 ;;
esac
[[ "$PHYSICAL_GPU" =~ ^[0-9]+$ ]] || exit 64
RUN="$PROJECT/log/八模型对比/2026候选骨干_HAM_20261104/$CANDIDATE"
mkdir -p "$RUN"
cd "$PROJECT"
exec 9>"$RUN/controller.lock"
flock -n 9 || exit 73
trap 'code=$?; printf "%s\n" "$code" > "$RUN/controller_exitcode"' EXIT
if [[ -e "$RUN/run/status.json" || -e "$RUN/run/checkpoints" ]]; then
  printf 'Refusing to overwrite existing training: %s\n' "$RUN"; exit 74
fi
GPU_FREE_MIB=$(nvidia-smi -i "$PHYSICAL_GPU" --query-gpu=memory.free --format=csv,noheader,nounits)
[[ "$GPU_FREE_MIB" =~ ^[0-9]+$ ]] || exit 75
if (( GPU_FREE_MIB < 8000 )); then
  printf 'GPU %s has %s MiB free; need 8000 MiB\n' "$PHYSICAL_GPU" "$GPU_FREE_MIB"; exit 75
fi
export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" EXPERIMENT_PHYSICAL_GPU="$PHYSICAL_GPU"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
if [[ "$CANDIDATE" == visionhope_tiny ]]; then
  export CPATH=/home/PuMengYu/.cache/skin_encoder2026/cuda121/nvidia/cuda_runtime/include:/home/PuMengYu/.cache/skin_encoder2026/cuda124_fixed/nvidia/cuda_cccl/include:/home/PuMengYu/anaconda3/envs/medseg/targets/x86_64-linux/include:/home/PuMengYu/anaconda3/envs/medseg/lib/python3.10/site-packages/nvidia/cusparse/include:/home/PuMengYu/anaconda3/envs/medseg/lib/python3.10/site-packages/nvidia/cusolver/include:/home/PuMengYu/anaconda3/envs/medseg/lib/python3.10/site-packages/nvidia/cublas/include
  export LIBRARY_PATH=/home/PuMengYu/.cache/skin_encoder2026/cuda121/nvidia/cuda_runtime/lib
  export CUDA_HOME=/home/PuMengYu/anaconda3/envs/medseg MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9
  export CC=/usr/bin/gcc-9 CXX=/usr/bin/g++-9
  unset NVCC_PREPEND_FLAGS CFLAGS CXXFLAGS CPPFLAGS CONDA_BUILD_SYSROOT
fi
printf '%s\n' "$$" > "$RUN/controller.pid"
"$PYTHON" -u source/comparison_2026/candidates_experiment.py --arm "$CANDIDATE" --mode preflight --run-root "$RUN" --gpu 0 > "$RUN/preflight_console.log" 2>&1
"$PYTHON" -u source/comparison_2026/candidates_experiment.py --arm "$CANDIDATE" --mode train --run-root "$RUN" --gpu 0 > "$RUN/training_console.log" 2>&1
"$PYTHON" -u source/comparison_2026/final_test_candidates.py --arm "$CANDIDATE" --gpu 0 > "$RUN/final_test_console.log" 2>&1
"$PYTHON" -u source/comparison_2026/update_candidates.py

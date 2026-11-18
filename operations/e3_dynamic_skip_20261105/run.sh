#!/bin/bash
set -euo pipefail
cd /home/PuMengYu/experiments/skin_segmentation
export CUDA_VISIBLE_DEVICES=0 EXPERIMENT_PHYSICAL_GPU=0 PYTHONUNBUFFERED=1
/home/PuMengYu/anaconda3/envs/medseg/bin/python -u source/e3_dynamic_skip/controller.py 2>&1 | tee -a log/TinyViM_E3.1_FDConvSkip_HAM_20261105/tmux_console.log

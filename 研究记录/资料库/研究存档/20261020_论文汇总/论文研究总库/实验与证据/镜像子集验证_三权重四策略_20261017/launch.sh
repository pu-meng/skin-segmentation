#!/usr/bin/env bash
set -euo pipefail
cd '/home/PuMengYu/论文汇总/论文研究总库/实验与证据/镜像子集验证_三权重四策略_20261017'
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/PuMengYu/nnUNet CUDA_VISIBLE_DEVICES=""
/home/PuMengYu/anaconda3/envs/medseg/bin/python -u wait_queue.py 2>&1 | tee -a queue.log

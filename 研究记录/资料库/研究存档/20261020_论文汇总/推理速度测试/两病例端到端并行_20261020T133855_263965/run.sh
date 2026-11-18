#!/bin/bash
set -eu
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/PuMengYu/nnUNet PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd '/home/PuMengYu/论文汇总/推理速度测试/两病例端到端并行_20261020T133855_263965'
exec /home/PuMengYu/anaconda3/envs/medseg/bin/python -u '/home/PuMengYu/论文汇总/推理速度测试/两病例端到端并行_20261020T133855_263965/code/benchmark.py'

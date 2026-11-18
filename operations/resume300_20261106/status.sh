#!/bin/bash
set -u
date -Is
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
nvidia-smi --query-gpu=index,uuid,memory.used,memory.free,utilization.gpu --format=csv,noheader
for arm in R0 R1 R2 R3 DualA; do
 d=/home/PuMengYu/experiments/skin_segmentation/operations/resume300_20261106/$arm
 echo "=== $arm ==="
 [ ! -f "$d/controller.pid" ] || ps -p "$(< "$d/controller.pid")" -o pid,ppid,sid,etime,args
 for f in controller_status.json preflight_passed.json train_identity.json first_resumed_optimizer_update.json failure.json completion.json; do [ ! -f "$d/$f" ] || { echo "$f"; cat "$d/$f"; }; done
 [ ! -f "$d/train.log" ] || { stat -c '%y %s %n' "$d/train.log"; tail -n 3 "$d/train.log"; }
done

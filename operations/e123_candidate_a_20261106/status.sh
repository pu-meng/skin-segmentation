#!/bin/bash
set -eu
R='/home/PuMengYu/experiments/skin_segmentation/log/TinyViM_E123_CandidateA_E2Spectral_B3x3_HAM100_20261106'
date -Is
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
ps -p "$(cat "$R/controller.pid")" -o user,pid,ppid,lstart,etime,args || true
nvidia-smi --query-gpu=index,uuid,memory.used,memory.free,utilization.gpu --format=csv,noheader
for f in controller_status.json preflight/preflight_status.json run/provenance.json run/status.json run/artifact_audit.json controller.exitcode; do
 if [ -f "$R/$f" ]; then stat -c '%n %y %s' "$R/$f"; cat "$R/$f"; fi
done
for f in preflight_console.log train_console.log; do
 if [ -f "$R/$f" ]; then stat -c '%n %y %s' "$R/$f"; tail -n 8 "$R/$f"; fi
done

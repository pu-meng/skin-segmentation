#!/bin/bash
set -eu
R=/home/PuMengYu/experiments/skin_segmentation
O=$R/operations/e31_fig2b_recovery_20261106
date --iso-8601=seconds
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ','); PIDS=${PIDS%,}
if [ -n "$PIDS" ]; then ps -p "$PIDS" -o user,pid,ppid,etime,args; fi
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,memory.free --format=csv,noheader
df -h "$R" /media/8T
for kind in e31 fig2b; do
 printf '\n=== %s controller ===\n' "$kind"
 if [ -f "$O/${kind}_status.json" ]; then cat "$O/${kind}_status.json"; fi
done
for a in E31-S1 fig2b42; do
 printf '\n=== %s progress ===\n' "$a"
 f="$O/${a}_resume_train_console.log"
 if [ -f "$f" ]; then stat -c '%y %s %n' "$f"; tail -4 "$f"; fi
done
for p in "$R/log/TinyViM_E3.1单Skip位置_HAM_20261106/E31-S1/run/status.json" "$R/log/mambalite_publiccode_four_seeds_20261106/fig2b/seed42/status.json"; do
 if [ -f "$p" ]; then printf '\n'; cat "$p"; fi
done
for p in "$R/log/TinyViM_E3.1单Skip位置_HAM_20261106/E31-S1/run/checkpoints/latest.pth" "$R/Models/mambalite_publiccode_four_seeds_20261106/fig2b/seed42/latest.pth" "$R/log/TinyViM_E3.1单Skip位置_HAM_20261106/E31-S1/run/artifact_audit.json" "$R/log/TinyViM_E3.1单Skip位置_HAM_20261106/E31-S2/run/artifact_audit.json" "$R/log/TinyViM_Decoder直接选择Skip_HAM_20261106/DG-S0/run/artifact_audit.json"; do
 if [ -f "$p" ]; then stat -c '%y %s %n' "$p"; else printf 'MISSING (not complete): %s\n' "$p"; fi
done

#!/bin/bash
set -eu
RUN=/home/PuMengYu/experiments/skin_segmentation/log/TinyViM_E1232_FourArm_SCREEN100of300_20261106
date -Is
for ARM in R0 R1 R2 R3; do
 echo "=== $ARM ==="
 for KIND in controller preflight train; do
  P="$RUN/$ARM/$KIND.pid"
  if test -f "$P"; then ps -p "$(cat "$P")" -o user,pid,ppid,lstart,etime,args || true; fi
 done
 for F in "$RUN/$ARM/controller_status.json" "$RUN/$ARM/controller.exitcode" "$RUN/$ARM/preflight/preflight.json" "$RUN/$ARM/train/status.json"; do
  if test -f "$F"; then stat -c '%y %s %n' "$F"; cat "$F"; fi
 done
 for F in "$RUN/$ARM/preflight_console.log" "$RUN/$ARM/train_console.log"; do
  if test -f "$F"; then stat -c '%y %s %n' "$F"; tail -n 4 "$F"; fi
 done
 for F in "$RUN/$ARM/train/checkpoints/latest.pth" "$RUN/$ARM/train/checkpoints/best.pth" "$RUN/$ARM/train/history.csv"; do
  if test -f "$F"; then stat -c '%y %s %n' "$F"; else echo "NOT_YET $F"; fi
 done
done
nvidia-smi --id=0 --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
for F in "$RUN/controller_status.json" "$RUN/FAILED.json"; do if test -f "$F"; then cat "$F"; fi; done

#!/bin/bash
set -u
OPS=/home/PuMengYu/experiments/skin_segmentation/operations/a234_resume300_20261107
RUN=/media/8T/PuMengYu/experiments/skin_segmentation/A234_RESUME100to300_20261107
date -Is
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | paste -sd,)
[ -z "$pids" ] || ps -p "$pids" -o pid,ppid,sid,etime,args
nvidia-smi --query-gpu=index,uuid,memory.used,memory.free,utilization.gpu --format=csv,noheader
for f in queue_status.json failure_controller.json launcher_exit.json; do [ ! -f "$OPS/$f" ] || { echo "$f"; cat "$OPS/$f"; }; done
for a in A2 A3 A4; do
 echo "=== $a ==="
 for f in cpu_roundtrip.json preflight_passed.json first_resumed_optimizer_update.json completion.json; do [ ! -f "$OPS/$a/$f" ] || { echo "$f"; cat "$OPS/$a/$f"; }; done
 for f in "$OPS/$a/preflight.log" "$OPS/$a/train.log" "$OPS/$a/test.log"; do [ ! -f "$f" ] || { stat -c '%y %s %n' "$f"; tail -n 3 "$f"; }; done
 for f in "$RUN/$a/run/status.json" "$RUN/$a/run/checkpoints/latest.pth" "$RUN/$a/final_test/none/summary.json" "$RUN/$a/final_test/mirror4/summary.json"; do [ ! -f "$f" ] || stat -c '%y %s %n' "$f"; done
done
df -h /home/PuMengYu /media/8T

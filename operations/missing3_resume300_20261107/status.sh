#!/usr/bin/env bash
set -euo pipefail
cd /home/PuMengYu/experiments/skin_segmentation
O=operations/missing3_resume300_20261107
R=/media/8T/PuMengYu/experiments/skin_segmentation/MISSING3_RESUME100to300_20261107
date --iso-8601=seconds
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | paste -sd, -)
if [ -n "$PIDS" ]; then ps -p "$PIDS" -o user,pid,ppid,lstart,etime,args; fi
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader
cat "$O/queue_status.json"
stat "$O/controller.log"
tail -n 8 "$O/controller.log"
for J in AFull ANoE2 DNoE2; do
 echo "$J"
 for F in cpu_restore_audit.json cpu_roundtrip.json preflight_passed.json first_resumed_optimizer_update.json completion.json; do if [ -f "$O/$J/$F" ]; then stat -c '%n %s %y' "$O/$J/$F"; else echo "MISSING $O/$J/$F"; fi; done
 for F in run/history.csv run/checkpoints/latest.pth run/checkpoints/best.pth final_test/none/summary.json final_test/mirror4/summary.json; do if [ -f "$R/$J/$F" ]; then stat -Lc '%n %s %y' "$R/$J/$F"; else echo "MISSING $R/$J/$F"; fi; done
done
df -h /home/PuMengYu /media/8T

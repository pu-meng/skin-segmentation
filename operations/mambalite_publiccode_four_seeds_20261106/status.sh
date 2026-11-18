#!/bin/bash
set -euo pipefail
ROOT=/home/PuMengYu/experiments/skin_segmentation
NAME=mambalite_publiccode_four_seeds_20261106
cd "$ROOT"
date -Is
pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
if [ -n "$pids" ]; then ps -p "$pids" -o user,pid,ppid,lstart,etime,args; fi
nvidia-smi -i 1 --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader
df -h "$ROOT" /media/8T
if [ -f "operations/$NAME/controller.pid" ]; then ps -p "$(cat "operations/$NAME/controller.pid")" -o user,pid,ppid,sid,lstart,etime,args; fi
for f in "log/$NAME/queue_status.json" "log/$NAME/cpu_acceptance.json" "log/$NAME/legacy_seed42_audit.json"; do if [ -f "$f" ]; then printf '\n--- %s ---\n' "$f"; cat "$f"; fi; done
for f in "operations/$NAME/controller_console.log" log/$NAME/public*/train_console.log "log/$NAME/fig2b/seed42/train_console.log"; do if [ -f "$f" ]; then stat -c '%y %s %n' "$f"; tail -4 "$f"; fi; done
for f in "operations/$NAME/protocol.json" "log/$NAME/reconstruction_resume_acceptance.json" "log/$NAME/report.md" "log/$NAME/results.json"; do if [ -f "$f" ]; then stat -c '%y %s %n' "$f"; else printf 'NOT_YET %s\n' "$f"; fi; done

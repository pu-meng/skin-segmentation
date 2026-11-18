#!/usr/bin/env bash
set -u
OUT='/home/PuMengYu/experiments/skin_segmentation/log/E123.1_低成本验证核查_20261105'
date --iso-8601=seconds
if [ -f "$OUT/pid" ]; then PID=$(cat "$OUT/pid"); ps -p "$PID" -o user,pid,ppid,lstart,etime,args; fi
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
for file in status.json gpu_preflight.json exitcode results.json audit.json report.md; do
 if [ -f "$OUT/$file" ]; then stat -c '%y %s %n' "$OUT/$file"; case "$file" in status.json|gpu_preflight.json|exitcode) cat "$OUT/$file";; esac; fi
done
if [ -f "$OUT/stdout.log" ]; then stat -c '%y %s %n' "$OUT/stdout.log"; tail -n 12 "$OUT/stdout.log"; fi
for group in full E2_off E3_off both_off E2_half E3_half; do printf '%s predictions=' "$group"; if [ -d "$OUT/$group/predictions" ]; then find "$OUT/$group/predictions" -maxdepth 1 -name '*.png' -printf '. ' | wc -w; else printf '0\n'; fi; done

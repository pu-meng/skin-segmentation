#!/usr/bin/env bash
set -u
O=/media/8T/PuMengYu/diagnostics/new_completed_bypass_20261106
date -Is
if [ -f "$O/pid" ]; then ps -p "$(<"$O/pid")" -o user,pid,ppid,lstart,etime,args; fi
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader
stat "$O/stdout.log"
tail -12 "$O/stdout.log"
for f in protocol.json cpu_preflight.json gpu_preflight.json partial_results.json results.json pixel_audit.json report.md status.json inference.exitcode exitcode; do
 if [ -f "$O/$f" ]; then stat -c '%n %s bytes %y' "$O/$f"; fi
done
if [ -f "$O/status.json" ]; then cat "$O/status.json"; fi

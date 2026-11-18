#!/usr/bin/env bash
set -euo pipefail
R=/media/8T/PuMengYu/experiments/skin_segmentation/A1_SkipC_v05_FRESH300_20261107
date --iso-8601=seconds
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | paste -sd, -)
if [ -n "$PIDS" ]; then ps -p "$PIDS" -o user,pid,ppid,lstart,etime,args; fi
if [ -f "$R/controller.pid" ]; then ps -p "$(<"$R/controller.pid")" -o user,pid,ppid,lstart,etime,args || true; fi
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader
for f in controller.log preflight.log train.log test.log; do if [ -f "$R/$f" ]; then stat "$R/$f"; tail -n 8 "$R/$f"; fi; done
/home/PuMengYu/anaconda3/envs/medseg/bin/python -B -c 'import pathlib,json; r=pathlib.Path("/media/8T/PuMengYu/experiments/skin_segmentation/A1_SkipC_v05_FRESH300_20261107"); files=["controller_status.json","cpu_acceptance.json","preflight/acceptance.json","first_optimizer_update.json","run/status.json","run/history.csv","run/checkpoints/best.pth","run/artifact_audit.json","final_test/none/summary.json","final_test/mirror4/summary.json","completion.json","failure.json"]; [(print(f,"exists",(r/f).stat().st_size,(r/f).stat().st_mtime) if (r/f).exists() else print(f,"MISSING")) for f in files]; print((r/"controller_status.json").read_text() if (r/"controller_status.json").exists() else "no controller state")'
df -h "$R" /home/PuMengYu

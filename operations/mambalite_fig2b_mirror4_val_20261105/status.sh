#!/bin/bash
set -u
R=/home/PuMengYu/experiments/skin_segmentation/log/mambalite_fig2b_mirror4_val_20261105
date --iso-8601=seconds
if test -f "$R/pid"; then ps -p "$(<"$R/pid")" -o user,pid,ppid,lstart,etime,args; fi
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader
stat "$R/stdout.log"
/home/PuMengYu/anaconda3/envs/medseg/bin/python -c 'from pathlib import Path; import json; r=Path("/home/PuMengYu/experiments/skin_segmentation/log/mambalite_fig2b_mirror4_val_20261105"); print("\n".join((r/"stdout.log").read_text().splitlines()[-8:])); print((r/"status.json").read_text()); [(print(p.relative_to(r), p.exists(), p.stat().st_size if p.exists() else None)) for p in [r/"gpu_preflight.json",r/"none/summary.json",r/"mirror4/summary.json",r/"results.json",r/"report.md",r/"audit.json",r/"exitcode"]]; print("PNG_COUNTS",{s:len(list((r/s/"predictions").glob("*.png"))) for s in ["none","mirror4"]})'

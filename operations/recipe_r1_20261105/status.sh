#!/usr/bin/env bash
set -euo pipefail
date -Is
ids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
if [ -n "$ids" ]; then ps -p "$ids" -o user,pid,ppid,lstart,etime,args; fi
nvidia-smi -i 0 --query-gpu=index,uuid,utilization.gpu,memory.used,memory.free,memory.total --format=csv,noheader
/home/PuMengYu/anaconda3/envs/medseg/bin/python -B - <<'PY'
from pathlib import Path
import json,datetime
q=Path('/home/PuMengYu/experiments/skin_segmentation/log/TinyViM_R1统一配方_HAM_20261105');o=Path('/home/PuMengYu/experiments/skin_segmentation/operations/recipe_r1_20261105')
for p in [q/'queue_status.json',o/'queue_console.log']:
 if p.exists():
  s=p.stat();print(str(p),'mtime',datetime.datetime.fromtimestamp(s.st_mtime).astimezone().isoformat(),'bytes',s.st_size)
  print(p.read_text()[-8000:])
for a in ['R1-E0','R1-B3','R1-B1']:
 r=q/a;print('ARM',a)
 for rel in ['protocol.json','cpu_check.json','prepare.exitcode','controller_status.json','preflight/preflight_status.json','run/status.json','run/history.csv','run/checkpoints/best.pth','run/artifact_audit.json','final_test/'+a+'/artifact_audit.json']:
  p=r/rel
  if p.exists():
   s=p.stat();print(rel,'EXISTS',s.st_size,datetime.datetime.fromtimestamp(s.st_mtime).astimezone().isoformat())
   if rel in ['controller_status.json','run/status.json']:print(p.read_text())
  else:print(rel,'ABSENT')
PY

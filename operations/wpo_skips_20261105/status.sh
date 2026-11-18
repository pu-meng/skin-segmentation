#!/bin/bash
set -euo pipefail
cd /home/PuMengYu/experiments/skin_segmentation
date --iso-8601=seconds
pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | paste -sd,)
if [ -n "$pids" ]; then ps -p "$pids" -o user,pid,ppid,etime,args; fi
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader
python3 - <<'PY'
import json,pathlib,datetime
q=pathlib.Path('log/TinyViM_WPO单Skip位置_HAM_20261105')
for p in [q/'queue_status.json']+[q/a/'controller_status.json' for a in ['WPO-S0','WPO-S1','WPO-S2']]:
 print(str(p),p.read_text() if p.exists() else 'NOT STARTED')
for a in ['WPO-S0','WPO-S1','WPO-S2']:
 r=q/a;print('\nARM',a)
 for rel in ['cpu_console.log','preflight_console.log','train_console.log','test_console.log']:
  p=r/rel
  if p.exists():print(str(p),'mtime',datetime.datetime.fromtimestamp(p.stat().st_mtime).isoformat(),'size',p.stat().st_size,'\n'+'\n'.join(p.read_text(errors='replace').splitlines()[-3:]))
 for rel in ['cpu_check.json','protocol.json','preflight/preflight_status.json','run/status.json','run/history.csv','run/checkpoints/best.pth','run/artifact_audit.json','final_test/'+a+'/artifact_audit.json']:
  p=r/rel
  print(rel,{'exists':p.exists(),'mtime':p.stat().st_mtime if p.exists() else None,'bytes':p.stat().st_size if p.exists() else None})
  if p.exists() and rel=='run/status.json':print(p.read_text())
PY

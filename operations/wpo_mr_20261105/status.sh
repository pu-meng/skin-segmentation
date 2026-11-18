#!/bin/bash
set -euo pipefail
cd /home/PuMengYu/experiments/skin_segmentation
date --iso-8601=seconds
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | paste -sd, -)
if [ -n "$pids" ]; then ps -p "$pids" -o user,pid,ppid,lstart,etime,args; fi
nvidia-smi -i 0 --query-gpu=index,uuid,memory.used,memory.free,utilization.gpu --format=csv,noheader
df -h '/home/PuMengYu/experiments/skin_segmentation'
/home/PuMengYu/anaconda3/envs/medseg/bin/python -B - <<'PY'
from pathlib import Path
import json,datetime
q=Path('/home/PuMengYu/experiments/skin_segmentation/log/TinyViM_B-WPO-MR_HAM_20261105');r=q/'B-WPO-MR'
for p in [q/'queue_status.json',r/'controller_status.json',r/'cpu_check.json',r/'preflight/preflight_status.json',r/'run/status.json']:
 if p.exists():
  d=json.loads(p.read_text())
  if p.name=='cpu_check.json':d={'passed':d['passed'],'checks':len(d['checks']),'parameters':d['parameters'],'static_c_numel':d['static_c_numel']}
  print(str(p),json.dumps(d,ensure_ascii=False))
 else:print(str(p),'NOT_CREATED')
for p in [q/'queue_console.log',r/'tmux_console.log',r/'train_console.log']:
 if p.exists():
  st=p.stat();print('LOG',p,'mtime',datetime.datetime.fromtimestamp(st.st_mtime).astimezone().isoformat(),'bytes',st.st_size)
  print(''.join(p.read_text(errors='replace').splitlines(True)[-6:]))
for rel in ['protocol.json','run/checkpoints/best.pth','run/artifact_audit.json','final_test/B-WPO-MR/artifact_audit.json','final_test/B-WPO-MR/test/summary.json','exitcode']:
 p=r/rel;print('ARTIFACT',p,dict(bytes=p.stat().st_size,mtime=p.stat().st_mtime) if p.exists() else 'NOT_CREATED')
PY

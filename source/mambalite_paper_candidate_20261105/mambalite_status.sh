#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/PuMengYu/experiments/skin_segmentation
LOG=$ROOT/log/mambalite_fig2b_20261105
date --iso-8601=seconds
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
/home/PuMengYu/anaconda3/envs/medseg/bin/python - <<'PY'
from pathlib import Path
import json,subprocess,datetime
r=Path('/home/PuMengYu/experiments/skin_segmentation/log/mambalite_fig2b_20261105')
p=r/'controller_status.json'
if p.exists():
 s=json.loads(p.read_text());print('CONTROLLER',json.dumps(s,ensure_ascii=False));pids=[str(s[k]) for k in ['pid','child_pid'] if s.get(k)]
 if pids:subprocess.run(['ps','-p',','.join(pids),'-o','user,pid,ppid,lstart,etime,args'])
for p in [r/'controller_console.log']+sorted(r.glob('seed*/status.json'))+sorted(r.glob('seed*/history.csv'))+sorted(r.glob('seed*/test/summary.json'))+[r/'five_run_summary.json']:
 if p.exists():
  st=p.stat();print('ARTIFACT',str(p),'bytes',st.st_size,'mtime',datetime.datetime.fromtimestamp(st.st_mtime).isoformat())
  if p.suffix in ['.log','.csv']:
   with p.open() as f:
    from collections import deque
    print(''.join(deque(f,maxlen=5)))
  else:print(p.read_text())
PY

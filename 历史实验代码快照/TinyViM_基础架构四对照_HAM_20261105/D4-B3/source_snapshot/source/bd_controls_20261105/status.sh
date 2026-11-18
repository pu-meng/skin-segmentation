#!/bin/bash
set -euo pipefail
date --iso-8601=seconds
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader
/home/PuMengYu/anaconda3/envs/medseg/bin/python -B - <<'PY'
import json,csv,subprocess
from pathlib import Path
q=Path('/home/PuMengYu/experiments/skin_segmentation/log/TinyViM_基础架构四对照_HAM_20261105')
for p in [q/'queue_status.json']+[q/a/'controller_status.json' for a in ['D4-B1','D4-B3','D4-BD','D4-B3D']]:
 print(str(p),p.read_text() if p.exists() else 'NOT_STARTED')
 if p.exists():
  s=json.loads(p.read_text());pid=s.get('child_pid',s.get('controller_pid'))
  if pid:subprocess.run(['ps','-p',str(pid),'-o','user,pid,ppid,lstart,etime,args'])
for a in ['D4-B1','D4-B3','D4-BD','D4-B3D']:
 d=q/a
 for rel in ['cpu_check.json','preflight/preflight_status.json','run/status.json','run/history.csv','run/checkpoints/best.pth','run/artifact_audit.json','final_test/'+a+'/artifact_audit.json']:
  p=d/rel;print(a,rel,dict(size=p.stat().st_size,mtime=p.stat().st_mtime) if p.exists() else 'MISSING')
 p=d/'run/history.csv'
 if p.exists():
  rows=list(csv.DictReader(p.open()));print(a,'COMPLETE_EPOCHS',len(rows),'RECENT',rows[-10:])
 for rel in ['cpu_console.log','preflight_console.log','train_console.log','test_console.log']:
  p=d/rel
  if p.exists():print(a,rel,'mtime',p.stat().st_mtime,'\n'+'\n'.join(p.read_text(errors='replace').splitlines()[-4:]))
PY

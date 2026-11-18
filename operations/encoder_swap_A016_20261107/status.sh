#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
/home/PuMengYu/anaconda3/envs/medseg/bin/python -B - <<'PY'
import json,subprocess,datetime
from pathlib import Path
r=Path('/media/8T/PuMengYu/experiments/skin_segmentation/EncoderSwap_A0_A1_A6_FRESH300_20261107')
print('TIME',datetime.datetime.now().astimezone().isoformat())
print(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True))
s=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True);pids=[s.strip() for s in s.splitlines() if s.strip().isdigit()]
if pids:print(subprocess.check_output(['ps','-p',','.join(pids),'-o','user,pid,ppid,lstart,etime,args'],text=True))
print(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True))
for name in ['controller_status.json','prepare_complete.json','storage_budget.json']:
 p=r/name
 if p.exists():print(name,p.read_text())
p=r/'protocol.json'
if p.exists():
 for job in json.loads(p.read_text())['jobs']:
  root=r/job;print('ARM',job)
  for name in ['cpu_acceptance.json','preflight/acceptance.json','first_optimizer_update.json','completion.json','final_test/none/artifact_audit.json','final_test/mirror4/artifact_audit.json']:
   q=root/name
   print(name,'MISSING' if not q.exists() else f'bytes={q.stat().st_size} mtime={datetime.datetime.fromtimestamp(q.stat().st_mtime).astimezone().isoformat()}')
  for name in ['preflight.log','train.log','test.log']:
   q=root/name
   if q.exists():
    print(name,'mtime',datetime.datetime.fromtimestamp(q.stat().st_mtime).astimezone().isoformat(),'bytes',q.stat().st_size)
    with q.open('rb') as f:f.seek(max(0,q.stat().st_size-1600));print(f.read().decode(errors='replace'))
PY

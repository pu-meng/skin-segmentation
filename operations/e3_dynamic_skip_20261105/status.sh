#!/bin/bash
set -euo pipefail
ROOT=/home/PuMengYu/experiments/skin_segmentation/log/TinyViM_E3.1_FDConvSkip_HAM_20261105
date --iso-8601=seconds
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
/home/PuMengYu/anaconda3/envs/medseg/bin/python - "$ROOT" <<'PY'
from pathlib import Path
import sys,json,subprocess,datetime
r=Path(sys.argv[1])
for n in ['controller_status.json','run/status.json','preflight/preflight_status.json','test_status.json']:
 p=r/n
 if p.exists():
  d=json.loads(p.read_text());print(n,json.dumps(d,ensure_ascii=False))
  if n=='controller_status.json':
   subprocess.run(['ps','-p',str(d['controller_pid'])+','+str(d.get('child_pid',0)),'-o','user,pid,ppid,lstart,etime,args'])
for n in ['preflight_console.log','train_console.log','test_console.log','run/history.csv','run/checkpoints/best.pth','run/checkpoints/latest.pth','run/checkpoints/final.pth','test/none/summary.json','test/mirror4/summary.json','results.json']:
 p=r/n
 if p.exists():
  print(n,p.stat().st_size,datetime.datetime.fromtimestamp(p.stat().st_mtime).isoformat())
  if p.suffix=='.log':print(p.read_text()[-1800:])
PY

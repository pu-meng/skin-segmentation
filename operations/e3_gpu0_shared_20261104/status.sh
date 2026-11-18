#!/usr/bin/env bash
set -eu
cd /home/PuMengYu/experiments/skin_segmentation
date --iso-8601=seconds
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
/home/PuMengYu/anaconda3/envs/medseg/bin/python - <<'CHECK'
from pathlib import Path
import json,subprocess
r=Path('log/TinyViM_三模块_E3_FDCG_HAM_20261104')
for n in ['controller_status.json','preflight/preflight_status.json','run/status.json']:
 p=r/n
 print(n,p.read_text() if p.exists() else 'absent')
s=json.loads((r/'controller_status.json').read_text())
subprocess.run(['ps','-p',','.join(str(x) for x in [s['controller_pid'],s.get('child_pid',0),2089491,2090011]),'-o','user,pid,ppid,etime,args'])
for n in ['train_console.log','run/history.csv','run/checkpoints/latest.pth','run/checkpoints/best.pth','final_test/E3/artifact_audit.json']:
 p=r/n
 print(n,dict(size=p.stat().st_size,mtime=p.stat().st_mtime) if p.exists() else 'absent')
 if p.exists() and p.suffix in ['.log','.csv']:print('\n'.join(p.read_text().splitlines()[-4:]))
CHECK

#!/usr/bin/env bash
set -euo pipefail
python - <<'PYCODE'
import csv,datetime,json,pathlib,subprocess
root=pathlib.Path('/home/PuMengYu/experiments/skin_segmentation/log')
print('TIME',datetime.datetime.now().astimezone().isoformat(),flush=True)
subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'])
for name,session in [('瓶颈Mamba_版本1','mamba_v1_gpu0'),('瓶颈Mamba_版本2','mamba_v2_gpu0'),('UNet2D_瓶颈Mamba_版本1','unet_mamba_v1_gpu1'),('UNet2D_瓶颈Mamba_版本2','unet_mamba_v2_gpu1')]:
 b=root/name;p=b/'run/status.json'
 print('\nARM',name,'TMUX',session,flush=True)
 if p.exists():
  d=json.loads(p.read_text());print(json.dumps({k:d.get(k) for k in ['status','pid','epoch','epoch_completed','step','steps_per_epoch','best_epoch','best_val_loss']},ensure_ascii=False),flush=True)
  if d.get('pid'):subprocess.run(['ps','-p',str(d['pid']),'-o','user,pid,ppid,etime,args'])
 for rel in ['preflight/failure.json','run/failure.json','controller_exitcode']:
  q=b/rel
  if q.exists():print(rel,q.read_text(),flush=True)
 h=b/'run/history.csv'
 if h.exists():
  rows=list(csv.DictReader(h.open()))
  print('LATEST_COMPLETED_EPOCH',json.dumps(rows[-1] if rows else {},ensure_ascii=False),flush=True)
 log=b/'training_console.log'
 if log.exists():print('LATEST_LOG','\n'.join(log.read_text(errors='replace').splitlines()[-2:]),flush=True)
PYCODE

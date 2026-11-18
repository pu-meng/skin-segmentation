#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
from pathlib import Path
import json,csv,datetime,statistics,subprocess
r=Path('/home/PuMengYu/experiments/skin_segmentation/log/UNet2D_瓶颈Mamba_版本2_Triton_正式300轮_20261030')
now=datetime.datetime.now().astimezone();print('TIME',now.isoformat(),flush=True)
for name in ['formal_status.json','formal_failure.json','run/status.json']:
 p=r/name
 if p.exists():
  d=json.loads(p.read_text());print(name,json.dumps({k:d.get(k) for k in ['status','pid','epoch','epoch_completed','step','steps_per_epoch','best_epoch','epochs','error']},ensure_ascii=False),flush=True)
  if name=='formal_status.json':subprocess.run(['ps','-p',str(d['pid']),'-o','user,pid,etime,args'])
p=r/'run/history.csv'
if p.exists():
 with p.open() as f:rows=list(csv.DictReader(f))
 if rows:
  sec=statistics.median(float(x['seconds']) for x in rows[-5:]);state=json.loads((r/'run/status.json').read_text());remaining=max(0,300-state['epoch_completed']-state.get('step',0)/state.get('steps_per_epoch',873))*sec
  print('RECENT_EPOCH_SECONDS',sec,'ESTIMATED_REMAINING_HOURS',remaining/3600,'ROUGH_FINISH',(now+datetime.timedelta(seconds=remaining)).isoformat())
p=r/'console.log'
if p.exists():print('LOG_END',chr(10).join(p.read_text().splitlines()[-3:]))
for name in ['run/checkpoints/latest.pth','run/checkpoints/best.pth','test/summary.json']:
 p=r/name
 if p.exists():print('ARTIFACT',name,'MTIME',datetime.datetime.fromtimestamp(p.stat().st_mtime).astimezone().isoformat())
p=r/'test/summary.json'
if p.exists():
 d=json.loads(p.read_text());print('TEST',json.dumps({'images':d['images'],'dice_pct':100*d['global']['dice'],'selected_epoch':d['selected_epoch']}))
PY

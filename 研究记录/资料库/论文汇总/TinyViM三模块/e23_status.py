"""Read-only E2/E3 remote status, outside frozen source manifest."""
from pathlib import Path
import subprocess,json
from datetime import datetime
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
print('TIME',datetime.now().astimezone().isoformat())
subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'])
subprocess.run(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total','--format=csv,noheader'])
for arm,label in [('E2','HSA'),('E3','FDCG')]:
 r=ROOT/f'log/TinyViM_三模块_{arm}_{label}_HAM_20261104'
 print('\nARM',arm, 'ROOT',r,flush=True)
 p=r/'controller_status.json'
 state=json.loads(p.read_text()) if p.exists() else {}
 print(json.dumps(state,ensure_ascii=False,indent=2),flush=True)
 pids=[str(state[k]) for k in ('controller_pid','child_pid') if state.get(k)]
 if pids:subprocess.run(['ps','-p',','.join(pids),'-o','user,pid,ppid,sid,lstart,etime,args'])
 for name in ['controller_console.log','gpu_acceptance_console.log','preflight_console.log','train_console.log','test_console.log']:
  f=r/name
  if f.exists():
   s=f.stat();print('LOG',f,'bytes',s.st_size,'mtime',datetime.fromtimestamp(s.st_mtime).astimezone().isoformat())
   with f.open('rb') as stream:
    stream.seek(max(0,s.st_size-4000));print(stream.read().decode(errors='replace'))
 for rel in ['preflight/preflight_status.json','run/status.json','run/artifact_audit.json','run/checkpoints/best.pth','run/checkpoints/final.pth',f'final_test/{arm}/artifact_audit.json']:
  f=r/rel
  if f.exists():
   print('ARTIFACT',f,'bytes',f.stat().st_size)
   if f.suffix=='.json':print(f.read_text())

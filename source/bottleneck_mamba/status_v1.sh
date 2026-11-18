#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/PuMengYu/experiments/skin_segmentation/log/瓶颈Mamba_版本1
python - "$ROOT" <<'PYCODE'
import datetime,json,pathlib,subprocess,sys
b=pathlib.Path(sys.argv[1])
print('TIME',datetime.datetime.now().astimezone().isoformat())
for command in [
 ['nvidia-smi','--query-gpu=index,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],
 ['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader']]:
 subprocess.run(command,check=False)
p=b/'run/status.json'
if p.exists():
 d=json.loads(p.read_text());print('TRAINING_STATUS',json.dumps(d,ensure_ascii=False))
 if d.get('pid'):subprocess.run(['ps','-p',str(d['pid']),'-o','user,pid,ppid,lstart,etime,args'],check=False)
for rel in ['training_console.log','run/history.csv']:
 p=b/rel
 if p.exists():
  print('FILE',p,'MODIFIED',datetime.datetime.fromtimestamp(p.stat().st_mtime).astimezone().isoformat())
  print('\n'.join(p.read_text(errors='replace').splitlines()[-8:]))
for rel in ['preflight/preflight_status.json','run/checkpoints/latest.pth','run/checkpoints/best.pth',
            'run/checkpoints/final.pth','run/artifact_audit.json','run/failure.json','controller_exitcode']:
 p=b/rel
 if p.exists():print('ARTIFACT',rel,'BYTES',p.stat().st_size)
PYCODE

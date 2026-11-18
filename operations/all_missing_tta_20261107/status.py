"""Read-only exact queue/worker/artifact status."""
import json,subprocess,time
from pathlib import Path
R=Path('/home/PuMengYu/experiments/skin_segmentation');O=R/'operations/all_missing_tta_20261107';m=json.loads((O/'manifest.json').read_text());s=json.loads((O/'queue_status.json').read_text())
print(time.strftime('%Y-%m-%d %H:%M:%S %z'));print(json.dumps(s,ensure_ascii=False,indent=2))
pids=[str(s['pid'])]+([str(s['child_pid'])] if s.get('child_pid') else [])
subprocess.run(['ps','-p',','.join(pids),'-o','user,pid,ppid,sid,lstart,etime,args'])
subprocess.run(['nvidia-smi','-i',m['gpu_uuid'],'--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader'])
subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'])
for e in m['rows']:
 work=Path(e['output']).parent;p=work/'status.json'
 st=json.loads(p.read_text()) if p.exists() else {'status':e['status']}
 print(e['id'],e['name'],json.dumps(st,ensure_ascii=False))
 if e['status']=='complete':
  for f in [Path(e['output'])/'summary.json',Path(e['output'])/'per_image_metrics.csv',work/'artifact_audit.json']:
   a=f.stat();print('ARTIFACT',f,a.st_size,a.st_mtime)
log=O/'controller_console.log'
print('LOG',log,log.stat().st_mtime,log.stat().st_size);print('\n'.join(log.read_text()[-8000:].splitlines()[-16:]))

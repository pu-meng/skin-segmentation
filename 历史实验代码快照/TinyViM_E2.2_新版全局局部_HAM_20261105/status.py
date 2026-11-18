from pathlib import Path
import json,subprocess,datetime
R=Path(__file__).parent
print(datetime.datetime.now().astimezone().isoformat())
s=json.loads((R/'controller_status.json').read_text());print(json.dumps(s,ensure_ascii=False,indent=2))
subprocess.run(['ps','-p',','.join(str(s[k]) for k in ['controller_pid','child_pid'] if k in s),'-o','user,pid,ppid,lstart,etime,args'])
subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'])
subprocess.run(['nvidia-smi','-i','GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader'])
p=Path(s['log'])
if p.exists():print('LOG',p,'mtime',p.stat().st_mtime,'bytes',p.stat().st_size);print(''.join(p.read_text().splitlines(True)[-12:]))
for rel in ['protocol.json','preflight/preflight_status.json','run/status.json','run/history.csv','epoch_timing.json','run/checkpoints/best.pth','run/checkpoints/latest.pth','run/checkpoints/final.pth','final_test/E2.2/artifact_audit.json']:
 p=R/rel;print('ARTIFACT',rel,dict(bytes=p.stat().st_size,mtime=p.stat().st_mtime) if p.exists() else 'missing')
 if p.exists() and rel in ['epoch_timing.json','run/status.json']:print(p.read_text())


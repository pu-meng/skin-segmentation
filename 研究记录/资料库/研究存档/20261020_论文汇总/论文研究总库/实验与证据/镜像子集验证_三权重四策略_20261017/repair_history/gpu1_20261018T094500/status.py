# Read-only status: no state writes, no imports of torch.
import os,subprocess,json,time
from datetime import datetime,timezone
from common import ROOT,read
from telemetry import sample,upstream
print('UTC',datetime.now(timezone.utc).isoformat())
g=sample(); print('GPU0',json.dumps(g))
pids=g['compute_pids']
if pids: print(subprocess.check_output(['ps','-p',','.join(map(str,pids)),'-o','user,pid,ppid,lstart,etime,args'],text=True))
state=read(ROOT/'queue_state.json') if (ROOT/'queue_state.json').exists() else {}
print('QUEUE',json.dumps(state,ensure_ascii=False))
if state.get('pid'):
    p=subprocess.run(['ps','-p',str(state['pid']),'-o','pid,ppid,lstart,etime,args'],capture_output=True,text=True); print(p.stdout)
print('UPSTREAM',json.dumps(upstream(),ensure_ascii=False))
for name in ['queue.log','cpu_preflight_architecture.log','cpu_preflight.json','manifest.json','queue_config.json','comparison.json']:
    f=ROOT/name
    if not f.exists(): print(name,'MISSING'); continue
    s=f.stat(); print(name,'mtime_utc',datetime.fromtimestamp(s.st_mtime,timezone.utc).isoformat(),'bytes',s.st_size)
    if name=='queue.log': print('\n'.join(f.read_text(errors='replace').splitlines()[-3:]))
if (ROOT/'manifest.json').exists():
    manifest=read(ROOT/'manifest.json'); count=0
    for item in manifest['order']:
        p=ROOT/item; preds=list((p/'predictions').glob('*.nii.gz')); a=read(p/'artifact_audit.json') if (p/'artifact_audit.json').exists() else {}
        actual={x.name.removesuffix('.nii.gz') for x in preds}
        valid=a.get('complete') is True and actual==set(manifest['cases']) and (p/'predictions/summary.json').is_file() and (p/'per_case.csv').is_file()
        count+=bool(valid)
        print(item,'predictions',len(preds),'complete',bool(valid),'run_log',str(p/'run.log'))
    print('VERIFIED expected=12 complete='+str(count))
print('CPU preflight is NOT real-CT success. Waiting is NOT GPU prediction started.')

import os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
UPSTREAM=[(ROOT/'log/TinyViM_WPO单Skip位置_HAM_20261105',a) for a in ['WPO-S0','WPO-S1','WPO-S2']]+[(ROOT/'log/TinyViM_B-WPO-MR_HAM_20261105','B-WPO-MR'),(ROOT/'log/TinyViM_B3_E1_E3.1_HAM_20261105','B3-E1-E3.1')]
def load(p):return json.loads(p.read_text())
def decide(flags):
 for i,f in enumerate(flags):
  if f=='failed':raise RuntimeError('upstream failed '+str(i))
 if all(f is True for f in flags):return True,None
 return False,next(i for i,f in enumerate(flags) if f is not True)
def complete(root,arm):
 p=root/'controller_status.json'
 if not p.exists():return False
 s=load(p)
 if s['status'] in ['failed','blocked']:raise RuntimeError('upstream failed '+arm+': '+str(s))
 for phase in ['preflight','train','test']:
  x=root/(phase+'.exitcode')
  if x.exists() and x.read_text().strip()!='0':raise RuntimeError('upstream phase failed '+arm+' '+phase)
 if s['status']!='complete':return False
 assert (root/'exitcode').read_text().strip()=='0'
 for phase in ['preflight','train','test']:assert (root/(phase+'.exitcode')).read_text().strip()=='0'
 a=load(root/'run/artifact_audit.json');assert a['training_complete'] and a['validation_artifacts_complete'] and a['epochs']==300
 a=load(root/'final_test'/arm/'artifact_audit.json');assert a['complete'] and a['prediction_count']==2020
 for key in ['controller_pid','child_pid']:
  pid=s.get(key)
  if pid and Path('/proc/'+str(pid)).exists():
   # PID reuse is handled by checking process command, not generic os.kill alone.
   cmd=Path('/proc/'+str(pid)+'/cmdline').read_bytes().replace(b'\x00',b' ').decode(errors='replace')
   if any(x in cmd for x in ['arm_controller','gpu_phase.py','source.recipe_r1_20261105.entry','source.wpo_skips_20261105.entry']):return False
 return True

def decision():
 flags=[]
 for q,a in UPSTREAM:
  p=q/'queue_status.json'
  if p.exists() and load(p).get('status') in ['blocked','failed']:raise RuntimeError('upstream queue failed '+str(q))
  flags.append(complete(q/a,a))
 ok,index=decide(flags)
 return ok,dict(upstream=[dict(arm=a,root=str(q/a),complete=f) for (q,a),f in zip(UPSTREAM,flags)],waiting_for=None if ok else UPSTREAM[index][1])

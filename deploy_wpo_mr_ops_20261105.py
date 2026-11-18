from pathlib import Path
import json,hashlib,ast,subprocess
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation');S=ROOT/'source/wpo_mr_20261105';O=ROOT/'operations/wpo_mr_20261105';Q=ROOT/'log/TinyViM_B-WPO-MR_HAM_20261105';R=Q/'B-WPO-MR'
PY='/home/PuMengYu/anaconda3/envs/medseg/bin/python';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
proof=json.loads((R/'cpu_check.json').read_text());protocol=json.loads((R/'protocol.json').read_text())
assert proof['passed'] and proof['source_sha256']=={p.name:sha(p) for p in S.glob('*.py')}
assert all(sha(ROOT/rel)==h and sha(R/'source_snapshot'/rel)==h for rel,h in protocol['source_sha256'].items())
for n in ['cpu','prepare']:(R/(n+'.exitcode')).write_text('0\n')
def put(name,t):
 p=O/name;assert not p.exists();p.write_text(t)
 if p.suffix=='.py':ast.parse(t)
 if p.suffix=='.sh':subprocess.run(['bash','-n',str(p)],check=True);p.chmod(0o755)
put('scheduling.py','''"""Read-only reservation contract. Pure decision helpers are fixture-testable."""
import json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
UP=ROOT/'log/TinyViM_WPO单Skip位置_HAM_20261105'
def load(p):return json.loads(p.read_text())
def alive(pid):
 try:os.kill(int(pid),0);return True
 except ProcessLookupError:return False

def complete(root,arm):
 status=load(root/'controller_status.json')
 assert status['status']!='failed',arm+' upstream failed; blocked without retry'
 if status['status']!='complete':return False
 assert (root/'exitcode').read_text().strip()=='0',arm+' nonzero or missing final exitcode'
 for phase in ['preflight','train','test']:assert (root/(phase+'.exitcode')).read_text().strip()=='0',arm+' phase exitcode'
 train=load(root/'run/artifact_audit.json');assert train['training_complete'] and train['epochs']==300 and train['validation_artifacts_complete']
 audit=load(root/'final_test'/arm/'artifact_audit.json');assert audit['complete'] and audit['prediction_count']==2020
 assert not alive(status['child_pid']),arm+' completed status but child still alive'
 return True

def decision(up=UP):
 q=load(up/'queue_status.json');assert q['status']!='blocked','upstream queue blocked: '+str(q)
 # Never steal the gap between S0/S1 completion and the S2 launch.
 for arm in ['WPO-S0','WPO-S1']:
  p=up/arm/'controller_status.json'
  if not p.exists() or not complete(up/arm,arm):return False,'waiting_S0_S1_full_train_test_contract'
 p=up/'WPO-S2/controller_status.json'
 if not p.exists():return False,'S2_reserved_not_started'
 s=load(p);assert s['status']!='failed','S2 failed; review required'
 if s['status']=='complete':
  return complete(up/'WPO-S2','WPO-S2'),'S2_complete_second_slot_available'
 if q['status']=='batch_active' and q.get('batch')==['WPO-S2'] and s['status'] in ['preflight','train','test']:
  assert alive(s['controller_pid']) and alive(s['child_pid']),'S2 stale active state'
  assert (up/'WPO-S2/prepare.exitcode').read_text().strip()=='0'
  return True,'S2_active_reserved_slot_plus_B-WPO-MR_second_slot'
 return False,'S2_reserved_stage_gap_or_not_yet_active'
''')
put('wait_queue.py','''import os,sys,json,time,subprocess,fcntl,shutil,hashlib
from pathlib import Path
from datetime import datetime
from scheduling import decision
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent;sys.path.insert(0,str(ROOT))
from source.wpo_mr_20261105.selection import ARM,RUN,QUEUE
PY='/home/PuMengYu/anaconda3/envs/medseg/bin/python';UUID='GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde'
os.environ.update(CUDA_VISIBLE_DEVICES=UUID,EXPERIMENT_PHYSICAL_GPU='0',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',MKL_THREADING_LAYER='GNU')
started=time.monotonic();deadline=started+120*3600

def write(**data):
 data.update(controller_pid=os.getpid(),updated_at=datetime.now().astimezone().isoformat(),deadline_hours=120,arm=ARM,gpu=0)
 p=QUEUE/'queue_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2,ensure_ascii=False));t.replace(p);print(json.dumps(data,ensure_ascii=False),flush=True)
def verify():
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 proof=json.loads((RUN/'cpu_check.json').read_text());assert proof['passed']
 assert proof['source_sha256']=={p.name:sha(p) for p in (ROOT/'source/wpo_mr_20261105').glob('*.py')},'CPU source drift'
 protocol=json.loads((RUN/'protocol.json').read_text())
 for rel,h in protocol['source_sha256'].items():assert sha(ROOT/rel)==h and sha(RUN/'source_snapshot'/rel)==h,'source/snapshot drift '+rel
 for phase in ['cpu','prepare']:assert (RUN/(phase+'.exitcode')).read_text().strip()=='0'
with (QUEUE/'queue.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(QUEUE/'queue.pid').write_text(str(os.getpid()))
 try:
  verify();write(status='verified_CPU_and_prepare_waiting_GPU0_reservation',GPU_preflight_started=False,formal_training_started=False)
  while True:
   assert time.monotonic()<deadline,'120h deadline; blocked without launch'
   ok,reason=decision()
   if not ok:
    write(status='waiting_existing_GPU0_reservation',reason=reason,GPU_preflight_started=False,formal_training_started=False);time.sleep(30);continue
   free=int(subprocess.check_output(['nvidia-smi','-i',UUID,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());disk=shutil.disk_usage(ROOT).free
   if free<7500 or disk<8*1024**3:
    write(status='waiting_resources',reason=reason,free_MiB=free,required_MiB=7500,disk_free_bytes=disk,required_disk_bytes=8*1024**3,GPU_preflight_started=False,formal_training_started=False);time.sleep(60);continue
   # Re-read the upstream contract immediately before launch; no lock modifications to old queue.
   ok,reason=decision()
   if not ok:continue
   verify();assert not (RUN/'controller_status.json').exists(),'refuse duplicate controller'
   write(status='launching',reason=reason,free_MiB=free,disk_free_bytes=disk)
   subprocess.run(['tmux','new-session','-d','-s','B-WPO-MR_HAM_gpu0','bash '+str(HERE/'arm.sh')],check=True)
   break
  grace=time.monotonic()+120
  while True:
   assert time.monotonic()<deadline,'120h overall deadline'
   p=RUN/'controller_status.json'
   if not p.exists():assert time.monotonic()<grace,'controller did not start';time.sleep(2);continue
   state=json.loads(p.read_text());assert state['status']!='failed',str(state)
   if state['status']=='complete':
    from scheduling import complete
    assert complete(RUN,ARM);write(status='complete',arm_state=state);(QUEUE/'queue.exitcode').write_text('0\\n');break
   write(status='arm_active',arm_state=state);time.sleep(30)
 except BaseException as e:
  write(status='blocked',error=repr(e));(QUEUE/'queue.exitcode').write_text('1\\n');raise
''')
put('run.sh',f'''#!/bin/bash
set -euo pipefail
cd {ROOT}
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
{PY} -B -u {O}/wait_queue.py 2>&1 | tee '{Q}/queue_console.log'
''')
put('arm.sh',f'''#!/bin/bash
set -euo pipefail
cd {ROOT}
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MKL_THREADING_LAYER=GNU
{PY} -B -u -m source.wpo_mr_20261105.arm_controller 2>&1 | tee '{R}/tmux_console.log'
''')
put('status.sh',f'''#!/bin/bash
set -euo pipefail
cd {ROOT}
date --iso-8601=seconds
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | paste -sd, -)
if [ -n "$pids" ]; then ps -p "$pids" -o user,pid,ppid,lstart,etime,args; fi
nvidia-smi -i 0 --query-gpu=index,uuid,memory.used,memory.free,utilization.gpu --format=csv,noheader
df -h '{ROOT}'
{PY} -B - <<'PY'
from pathlib import Path
import json,datetime
q=Path('{Q}');r=q/'B-WPO-MR'
for p in [q/'queue_status.json',r/'controller_status.json',r/'cpu_check.json',r/'preflight/preflight_status.json',r/'run/status.json']:
 if p.exists():
  d=json.loads(p.read_text())
  if p.name=='cpu_check.json':d={{'passed':d['passed'],'checks':len(d['checks']),'parameters':d['parameters'],'static_c_numel':d['static_c_numel']}}
  print(str(p),json.dumps(d,ensure_ascii=False))
 else:print(str(p),'NOT_CREATED')
for p in [q/'queue_console.log',r/'tmux_console.log',r/'train_console.log']:
 if p.exists():
  st=p.stat();print('LOG',p,'mtime',datetime.datetime.fromtimestamp(st.st_mtime).astimezone().isoformat(),'bytes',st.st_size)
  print(''.join(p.read_text(errors='replace').splitlines(True)[-6:]))
for rel in ['protocol.json','run/checkpoints/best.pth','run/artifact_audit.json','final_test/B-WPO-MR/artifact_audit.json','final_test/B-WPO-MR/test/summary.json','exitcode']:
 p=r/rel;print('ARTIFACT',p,dict(bytes=p.stat().st_size,mtime=p.stat().st_mtime) if p.exists() else 'NOT_CREATED')
PY
''')
put('test_scheduling.py','''"""Synthetic state-machine fixtures only, not experiment metrics."""
from pathlib import Path
import json,os
import scheduling as s
ROOT=Path(__file__).parent/'scheduler_fixtures';ROOT.mkdir(exist_ok=True)
s.alive=lambda pid:pid==os.getpid()
def put(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d))
def done(a):
 r=ROOT/a
 put(r/'controller_status.json',dict(status='complete',child_pid=-99999))
 for phase in ['preflight','train','test']:(r/(phase+'.exitcode')).write_text('0')
 (r/'exitcode').write_text('0')
 put(r/'run/artifact_audit.json',dict(training_complete=True,epochs=300,validation_artifacts_complete=True))
 put(r/'final_test'/a/'artifact_audit.json',dict(complete=True,prediction_count=2020))
put(ROOT/'queue_status.json',dict(status='batch_active',batch=['WPO-S0','WPO-S1']))
assert not s.decision(ROOT)[0]
done('WPO-S0');done('WPO-S1');assert not s.decision(ROOT)[0]
r=ROOT/'WPO-S2';put(r/'controller_status.json',dict(status='waiting_pair_prepare',controller_pid=os.getpid(),child_pid=os.getpid()));assert not s.decision(ROOT)[0]
(r/'prepare.exitcode').write_text('0');put(ROOT/'queue_status.json',dict(status='batch_active',batch=['WPO-S2']))
for phase in ['preflight','train','test']:
 put(r/'controller_status.json',dict(status=phase,controller_pid=os.getpid(),child_pid=os.getpid()));assert s.decision(ROOT)[0]
put(r/'controller_status.json',dict(status='failed'))
try:s.decision(ROOT);raise RuntimeError('failure accepted')
except AssertionError:pass
done('WPO-S2');assert s.decision(ROOT)[0]
put(r/'final_test/WPO-S2/artifact_audit.json',dict(complete=True,prediction_count=2019))
try:s.decision(ROOT);raise RuntimeError('incomplete audit accepted')
except AssertionError:pass
print('SCHEDULER_FIXTURE_PASS: waiting, stage-gap, S2 concurrent slot, complete, failed, wrong-count rejection; no real metrics')
''')
manifest=dict(arm='B-WPO-MR',source=str(S),operations=str(O),run=str(R),CPU_passed=True,CPU_checks=len(proof['checks']),parameters=proof['parameters'],protocol_sha256=sha(R/'protocol.json'),source_sha256={p.name:sha(p) for p in S.glob('*.py')},upstream='WPO-S0/S1 full completion; S2 actual active or complete; do not modify upstream queue',gpu=0,uuid='GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde',deadline_hours=120,free_memory_MiB=7500,disk_guard_bytes=8*1024**3)
put('manifest.json',json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))

# R1 retired by user: refuse further queue, preflight, train or test launches.
from pathlib import Path as _R1Path
_R1_CANCEL = _R1Path(__file__).resolve().parents[2] / 'log/TinyViM_R1统一配方_HAM_20261105/CANCELLED_BY_USER.json'
if _R1_CANCEL.exists():
    print('R1 cancelled by user; launch disabled.', flush=True)
    raise SystemExit(0)

import os,sys,json,time,subprocess,fcntl,shutil,hashlib
from pathlib import Path
from datetime import datetime
from scheduling import decision,complete
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent
QUEUE=ROOT/'log/TinyViM_R1统一配方_HAM_20261105';PY='/home/PuMengYu/anaconda3/envs/medseg/bin/python'
UUID='GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde';deadline=time.time()+120*3600
ARMS=['R1-E0','R1-B3','R1-B1']
def write(**data):
 data.update(controller_pid=os.getpid(),updated_at=datetime.now().astimezone().isoformat(),deadline_unix=deadline,deadline_hours=120,gpu=0,maximum_new_tasks=2)
 p=QUEUE/'queue_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2,ensure_ascii=False));t.replace(p);print(json.dumps(data,ensure_ascii=False),flush=True)
def verify():
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 for arm in ARMS:
  r=QUEUE/arm;proof=json.loads((r/'cpu_check.json').read_text());assert proof['passed']
  assert proof['source_sha256']=={p.name:sha(p) for p in (ROOT/'source/recipe_r1_20261105').glob('*.py')}
  assert (r/'prepare.exitcode').read_text().strip()=='0'
  for rel,h in json.loads((r/'protocol.json').read_text())['source_sha256'].items():assert sha(ROOT/rel)==h and sha(r/'source_snapshot'/rel)==h
with (QUEUE/'queue.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(QUEUE/'queue.pid').write_text(str(os.getpid()))
 try:
  verify()
  while True:
   assert time.time()<deadline,'120h deadline'
   ok,reason=decision()
   if ok:break
   write(status='waiting_prior_GPU0_reservations',reservation=reason,CPU_passed=ARMS,GPU_preflight_started=False,formal_training_started=False);time.sleep(30)
  for batch in [['R1-E0','R1-B3'],['R1-B1']]:
   while True:
    assert time.time()<deadline,'120h deadline'
    ok,reason=decision();assert ok,'prior reservation changed'
    free=int(subprocess.check_output(['nvidia-smi','-i',UUID,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    required=9000*len(batch)+1536;disk=shutil.disk_usage(ROOT).free
    if free>=required and disk>=8*1024**3:break
    write(status='waiting_resources',batch=batch,free_MiB=free,required_MiB=required,disk_free_bytes=disk,required_disk_bytes=8*1024**3);time.sleep(30)
   verify();write(status='launching_batch',batch=batch,free_MiB=free,per_task_allocator_MiB=9000,headroom_MiB=1536,disk_free_bytes=disk)
   for arm in batch:
    assert not (QUEUE/arm/'controller_status.json').exists(),'refuse duplicate arm launch'
    env=os.environ.copy();env.update(R1_DEADLINE=str(deadline),R1_PAIR=json.dumps(batch),R1_ARM=arm)
    (QUEUE/arm/'launch_context.json').write_text(json.dumps(dict(deadline=deadline,pair=batch)))
    subprocess.run(['tmux','new-session','-d','-s',arm+'_HAM_gpu0','bash '+str(HERE/(arm+'.sh'))],env=env,check=True)
   grace=time.time()+120
   while True:
    assert time.time()<deadline,'120h overall deadline'
    done=[]
    for arm in batch:
     p=QUEUE/arm/'controller_status.json'
     if not p.exists():assert time.time()<grace,'arm did not start';done.append(False);continue
     state=json.loads(p.read_text());assert state['status']!='failed',str(state)
     done.append(complete(QUEUE/arm,arm))
    if all(done):break
    write(status='batch_active',batch=batch);time.sleep(30)
  write(status='complete',arms=ARMS);(QUEUE/'queue.exitcode').write_text('0\n')
 except BaseException as e:
  write(status='blocked',error=repr(e));(QUEUE/'queue.exitcode').write_text('1\n');raise

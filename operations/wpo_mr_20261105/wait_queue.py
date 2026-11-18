import os,sys,json,time,subprocess,fcntl,shutil,hashlib
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
    assert complete(RUN,ARM);write(status='complete',arm_state=state);(QUEUE/'queue.exitcode').write_text('0\n');break
   write(status='arm_active',arm_state=state);time.sleep(30)
 except BaseException as e:
  write(status='blocked',error=repr(e));(QUEUE/'queue.exitcode').write_text('1\n');raise

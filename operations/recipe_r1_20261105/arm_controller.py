# R1 retired by user: refuse further queue, preflight, train or test launches.
from pathlib import Path as _R1Path
_R1_CANCEL = _R1Path(__file__).resolve().parents[2] / 'log/TinyViM_R1统一配方_HAM_20261105/CANCELLED_BY_USER.json'
if _R1_CANCEL.exists():
    print('R1 cancelled by user; launch disabled.', flush=True)
    raise SystemExit(0)

import os,sys,subprocess,json,time,fcntl,threading,shutil,hashlib
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from source.recipe_r1_20261105.selection import ARM,RUN,QUEUE
UUID='GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde'
os.environ.update(CUDA_VISIBLE_DEVICES=UUID,EXPERIMENT_PHYSICAL_GPU='0',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',MKL_THREADING_LAYER='GNU')
context=json.loads((RUN/'launch_context.json').read_text());DEADLINE=context['deadline'];PAIR=context['pair']
def write(name,value):
 p=RUN/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2,ensure_ascii=False));t.replace(p)
def verify():
 proof=json.loads((RUN/'cpu_check.json').read_text());assert proof['passed']
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 assert proof['source_sha256']=={p.name:sha(p) for p in (ROOT/'source/recipe_r1_20261105').glob('*.py')}
 for rel,h in json.loads((RUN/'protocol.json').read_text())['source_sha256'].items():assert sha(ROOT/rel)==h and sha(RUN/'source_snapshot'/rel)==h,'source drift '+rel

def snapshot(child,phase):
 while child.poll() is None:
  out=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
  matches=[line for line in out.splitlines() if line.split(',')[1].strip()==str(child.pid)]
  assert all(x.split(',')[0].strip()==UUID for x in matches),'wrong GPU'
  write(phase+'_gpu_snapshot.json',dict(pid=child.pid,uuid=UUID,matches=matches,all_compute=out))
  time.sleep(10)
with (RUN/'controller.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state=dict(arm=ARM,controller_pid=os.getpid(),gpu=0,gpu_uuid=UUID,deadline=DEADLINE)
 try:
  verify()
  for phase in ['preflight','train','test']:
   assert time.time()<DEADLINE,'120h overall deadline'
   assert shutil.disk_usage(ROOT).free>=8*1024**3,'disk below8GiB'
   if phase=='train':
    state.update(status='waiting_pair_preflight');write('controller_status.json',state)
    while True:
     for arm in PAIR:
      p=QUEUE/arm/'controller_status.json'
      if p.exists():assert json.loads(p.read_text())['status']!='failed','paired preflight failed'
     if all((QUEUE/arm/'preflight.exitcode').exists() and (QUEUE/arm/'preflight.exitcode').read_text().strip()=='0' for arm in PAIR):break
     assert time.time()<DEADLINE;time.sleep(5)
   verify()
   child=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'operations/recipe_r1_20261105/gpu_phase.py'),phase],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
   state.update(status=phase,child_pid=child.pid,updated_at=datetime.now().astimezone().isoformat());write('controller_status.json',state)
   print(ARM,phase,'PID',child.pid,flush=True)
   timer=threading.Timer(max(1,DEADLINE-time.time()),child.terminate);timer.daemon=True;timer.start()
   threading.Thread(target=snapshot,args=(child,phase),daemon=True).start()
   with (RUN/(phase+'_console.log')).open('x') as log:
    for line in child.stdout:print(line,end='',flush=True);log.write(line);log.flush()
   rc=child.wait();timer.cancel();(RUN/(phase+'.exitcode')).write_text(str(rc)+'\n');assert rc==0,f'{phase} failed {rc}; no retry'
   if phase=='preflight':
    p=json.loads((RUN/'preflight/preflight_status.json').read_text());assert p['status']=='passed' and p['optimizer_steps']==3
   if phase=='train':
    a=json.loads((RUN/'run/artifact_audit.json').read_text());assert a['training_complete'] and a['epochs']==300 and a['validation_artifacts_complete']
   if phase=='test':
    a=json.loads((RUN/'final_test'/ARM/'artifact_audit.json').read_text());assert a['complete'] and a['prediction_count']==2020
  state.update(status='complete');write('controller_status.json',state);(RUN/'exitcode').write_text('0\n')
 except BaseException as e:
  state.update(status='failed',error=repr(e));write('controller_status.json',state);(RUN/'exitcode').write_text('1\n');raise

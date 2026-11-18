import fcntl, json, os, signal, subprocess, sys, threading, time, traceback
from pathlib import Path

ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
OPS=Path(__file__).resolve().parent
GATE=ROOT/'operations/encoder_swap_3slot_takeover_20261108/admission.lock'
UUID0='GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde'
PY='/home/PuMengYu/anaconda3/envs/medseg/bin/python'
AOPS=ROOT/'operations/encoder_swap_3slot_takeover_20261108'
ARUN=Path('/media/8T/PuMengYu/experiments/skin_segmentation/EncoderSwap_A0_A1_A6_FRESH300_20261107')
AENTRY=ROOT/'operations/encoder_swap_A016_20261107/entry.py'
IOPS=ROOT/'operations/isic_a1_encoder_20261108'
IRUN=Path('/media/8T/PuMengYu/experiments/skin_segmentation/ISIC2018_A1_EncoderSwap_FRESH300_20261108')

TASKS=[
 dict(key='ham_mobileone_s1_A1',kind='ham',gpu1_worker=1275054,source='rmt_s__A1',target='mobileone_s1__A1'),
 dict(key='ham_mobileone_s1_A6',kind='ham',gpu1_worker=1275058,source='rmt_s__A6',target='mobileone_s1__A6'),
 dict(key='isic_a1_mobileone_s1',kind='isic',gpu1_worker=2525428,source='rmt_s',target='mobileone_s1'),
]

def write(p,o):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(o,ensure_ascii=False,indent=2));tmp.replace(p)
def read(p):return json.loads(Path(p).read_text())
def alive(pid):
 try:
  stat=Path(f'/proc/{int(pid)}/stat').read_text();return stat[stat.rindex(')')+2:].split()[0]!='Z'
 except (OSError,ValueError):return False
def proc_cmd(pid):
 try:return Path(f'/proc/{int(pid)}/cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
 except OSError:return ''
def children(pid):
 try:return [int(x) for x in Path(f'/proc/{pid}/task/{pid}/children').read_text().split()]
 except OSError:return []
def acquire():
 f=GATE.open('a');fcntl.flock(f,fcntl.LOCK_EX);return f
def release(f):fcntl.flock(f,fcntl.LOCK_UN);f.close()
def mark(task,**kw):
 p=OPS/(task['key']+'.json');o=read(p) if p.exists() else {'task':task['key']};o.update(kw,updated_unix=time.time());write(p,o)

def intercept(task):
 pid=task['gpu1_worker'];cmd=proc_cmd(pid)
 if not alive(pid) or ('operations/encoder_swap_3slot_takeover_20261108/controller.py worker 1:' not in cmd if task['kind']=='ham' else 'operations/isic_a1_encoder_20261108/run.py worker 1' not in cmd):
  raise RuntimeError(f"expected GPU1 worker {pid} missing or changed: {cmd}")
 if task['kind']=='ham':
  st=AOPS/('state_'+('1_1' if task['target'].endswith('A1') else '1_2')+'.json')
  marker=AOPS/(task['source']+'_test_exitcode.json')
  test_started=lambda: (read(st).get('job')==task['source'] and read(st).get('status')=='test')
 else:
  st=IRUN/'mobileone_s1'/'preflight_status.json'
  marker=IRUN/'rmt_s'/'test_exitcode.json'
  test_started=lambda: (IRUN/'rmt_s'/'test_status.json').exists()
 if st.exists() and task['kind']=='isic':raise RuntimeError('target ISIC arm already started; refusing duplicate/migration: '+str(st))
 mark(task,status='waiting_for_source_test',gpu1_worker=pid,source=task['source'],target=task['target'])
 while not test_started():
  if not alive(pid):raise RuntimeError(f'GPU1 worker {pid} exited before source test')
  time.sleep(.5)
 # Hold the shared phase-admission lock from the source test through its exit,
 # so its worker cannot start the queued MobileOne preflight on GPU1.
 lock=acquire();mark(task,status='holding_admission_until_source_test_exit')
 try:
  while not marker.exists():
   if not alive(pid):raise RuntimeError(f'GPU1 worker {pid} exited before recording source test completion')
   time.sleep(.25)
  if task['kind']=='ham':
   deadline=time.time()+60
   while True:
    d=read(st)
    if d.get('job')==task['target'] and d.get('status')=='queued' and d.get('child_pid') is None:break
    if not alive(pid):raise RuntimeError('HAM worker exited before queue handoff')
    if time.time()>deadline:raise TimeoutError('HAM slot did not reach queued handoff state')
    time.sleep(.1)
  else:
   if st.exists():raise RuntimeError('ISIC MobileOne preflight unexpectedly started on GPU1; no duplicate will be launched')
  live_children=[c for c in children(pid) if alive(c)]
  if live_children:raise RuntimeError(f'refusing to stop GPU1 worker with live child processes: {live_children}')
  os.kill(pid,signal.SIGTERM)
  limit=time.time()+10
  while alive(pid) and time.time()<limit:time.sleep(.1)
  if alive(pid):os.kill(pid,signal.SIGKILL)
  mark(task,status='rerouted_waiting_for_gpu0',former_gpu=1,target_gpu=0,worker_stopped_after_source_test=True,source_test_exitcode=read(marker).get('exitcode'))
 finally:release(lock)

def resource_ready():
 rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 pids=[int(x.split(',')[1]) for x in rows.splitlines() if x.startswith(UUID0)]
 stats=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader'],text=True)
 free=int(next(x.split(',')[1].strip().split()[0] for x in stats.splitlines() if x.startswith(UUID0)))
 return len(pids)<4 and free>=8192,pids,free

def launch(task,mode):
 while True:
  lock=acquire()
  ready,pids,free=resource_ready()
  if ready:break
  release(lock);mark(task,status='waiting_gpu0_headroom',mode=mode,compute_pids=pids,free_mib=free)
  time.sleep(30)
 if task['kind']=='ham':argv=[PY,'-B','-u',str(AENTRY),mode,task['target'],'0']
 else:argv=[PY,'-B','-u',str(IOPS/'run.py'),'phase',mode,task['target'],'0']
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=UUID0,EXPERIMENT_PHYSICAL_GPU='0',EXPERIMENT_GPU_UUID=UUID0,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 logfile=OPS/(task['key']+'_'+mode+'.log')
 with logfile.open('a',buffering=1) as log:
  child=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  mark(task,status=mode,mode=mode,gpu=0,child_pid=child.pid,started_unix=time.time())
  limit=time.time()+180
  while child.poll() is None:
   out=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
   if child.pid in [int(x.split(',')[1]) for x in out.splitlines() if x.startswith(UUID0)]:break
   if time.time()>limit:
    child.terminate();raise TimeoutError(f'{task["key"]} {mode} did not become resident on GPU0')
   time.sleep(1)
 release(lock)
 rc=child.wait();mark(task,status='phase_complete' if rc==0 else 'failed',mode=mode,exitcode=rc,finished_unix=time.time())
 if rc:raise RuntimeError(f'{task["key"]} {mode} exited {rc}; stopping reroute queue')

def run_task(task):
 if task['kind']=='ham':
  run=ARUN/task['target']; phases=['preflight','train','test']
 else:
  run=IRUN/task['target']; phases=['preflight','train','test']
 if (run/'completion.json').exists() and read(run/'completion.json').get('complete'):raise RuntimeError('target already complete; refusing to repeat')
 for mode in phases:
  if task['kind']=='ham':
   marker=AOPS/(task['target']+'_'+mode+'_exitcode.json')
  else:marker=run/(mode+'_exitcode.json')
  if marker.exists():raise RuntimeError(f'refusing duplicate phase because marker exists: {marker}')
  launch(task,mode)
 if task['kind']=='ham':
  c=read(run/'completion.json');assert c.get('complete') and c.get('epochs')==300
  for mode in ('none','mirror4'):assert read(run/'final_test'/mode/'artifact_audit.json').get('complete')
 else:assert read(run/'completion.json').get('complete')
 mark(task,status='complete',gpu=0,completion_verified=True,completed_unix=time.time())

def main():
 status=OPS/'migration_status.json'
 write(status,dict(status='intercepting_gpu1_queue',tasks=[x['key'] for x in TASKS],gpu0=0,started_unix=time.time()))
 try:
  errors=[]
  def guarded(task):
   try:intercept(task)
   except BaseException as e:errors.append((task['key'],e,traceback.format_exc()))
  threads=[]
  for t in TASKS:
   th=threading.Thread(target=guarded,args=(t,),daemon=True);th.start();threads.append(th)
  for th in threads:th.join()
  if errors:raise RuntimeError('handoff failed: '+'; '.join(k+': '+repr(e)+' '+tb for k,e,tb in errors))
  for t in TASKS:
   run_task(t)
  write(status,dict(status='complete',tasks=[x['key'] for x in TASKS],gpu0=0,completed_unix=time.time()))
 except BaseException as e:
  write(status,dict(status='failed',error=repr(e),traceback=traceback.format_exc(),updated_unix=time.time()))
  raise

if __name__=='__main__':main()

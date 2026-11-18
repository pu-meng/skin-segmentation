import os,sys,json,time,hashlib,fcntl,subprocess,signal,traceback
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
OPS=Path(__file__).resolve().parent
OLD=ROOT/'operations/encoder_swap_A016_20261107'
RUN=Path('/media/8T/PuMengYu/experiments/skin_segmentation/EncoderSwap_A0_A1_A6_FRESH300_20261107')
PY='/home/PuMengYu/anaconda3/envs/medseg/bin/python'
UUID=['GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde','GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b']
PLAN={'0:0':['rmt_t__A1','swiftformer_s__A0'],'0:1':['rmt_t__A6','swiftformer_s__A1'],'0:2':['rmt_s__A0','swiftformer_s__A6'],'1:0':['rmt_t__A0','mobileone_s1__A0'],'1:1':['rmt_s__A1','mobileone_s1__A1'],'1:2':['rmt_s__A6','mobileone_s1__A6']}
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.'+str(os.getpid())+'.tmp');t.write_text(json.dumps(v,indent=2,ensure_ascii=False));t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def verify():
 p=read(RUN/'protocol.json');assert read(RUN/'prepare_complete.json')['count']==12
 for f,h in p['protected_sha256'].items():assert sha(f)==h,'source drift '+f
 assert sha(p['config']['data']['split_file'])==p['config']['data']['split_sha256']
 for job in sum(PLAN.values(),[]):
  q=read(RUN/job/'protocol.json');a=read(RUN/job/'cpu_acceptance.json')
  assert q['parent_sha256']==sha(RUN/'protocol.json') and a['passed'] and a['initial_state_sha256']==q['initial_state_sha256']
 return p

def phases(adopt):return ['test'] if adopt else ['preflight','train','test']
def allowed(blocked,now,deadline):return not blocked and now<deadline
def fixture():
 jobs=sum(PLAN.values(),[]);assert len(jobs)==len(set(jobs))==12
 for g in [0,1]:assert len([k for k in PLAN if k.startswith(str(g)+':')])==3
 assert jobs.count('rmt_t__A0')==1 and PLAN['1:0'][0]=='rmt_t__A0'
 assert phases(True)==['test'] and 'train' not in phases(True)
 assert not allowed(True,0,1) and not allowed(False,1,1) and allowed(False,0,1)
 write(OPS/'fixture.json',dict(passed=True,unique_arms=12,slots_per_gpu=3,adoption_test_only=True,failure_blocks_new_phases=True,deadline_blocks_new_phases=True))

def proc(pid):
 text=Path('/proc/%d/stat'%pid).read_text();v=text[text.rindex(')')+2:].split();return dict(state=v[0],ppid=int(v[1]),pgrp=int(v[2]),session=int(v[3]),start_ticks=int(v[19]),exit_code=int(v[49]))
def train_audit(job):
 a=read(RUN/job/'run/artifact_audit.json');assert a['training_complete'] and a['validation_artifacts_complete'] and a['epochs']==300

def block(key,exc):
 v=dict(slot=key,error=repr(exc),traceback=traceback.format_exc(),time=time.time());write(OPS/('failure_'+key.replace(':','_')+'.json'),v)
 with (OPS/'admission.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);write(OPS/'blocked.json',v)

def adoption(state):
 expected=read(OPS/'takeover.json');pid=expected['child_pid']
 while True:
  p=proc(pid);assert p['start_ticks']==expected['child_start_ticks'];assert 'frozen 1' in Path(expected['freezer_events']).read_text()
  if p['state']=='Z':break
  if time.time()>=read(OPS/'manifest.json')['deadline_unix']:raise TimeoutError('adoption deadline: training left untouched; old controller remains stopped')
  time.sleep(15)
 write(OPS/'adopted_train_exit.json',dict(pid=pid,proc_stat=p,raw_wait_status=p['exit_code'],exitcode=os.waitstatus_to_exitcode(p['exit_code']),time=time.time()))
 assert os.waitstatus_to_exitcode(p['exit_code'])==0,'adopted training failed'
 train_audit('rmt_t__A0');verify()
 # The only direct phase child must now be a zombie. Never close a pane with live training.
 children=[int(v) for f in Path('/proc/%d/task'%expected['old_pid']).glob('*/children') for v in f.read_text().split()]
 assert set(children)=={pid} and proc(pid)['state']=='Z'
 os.kill(expected['old_pid'],signal.SIGKILL)
 # Reacquire original flock only after its owner has gone; retain for this adopted slot.
 lock=(RUN/'controller.lock').open('a');limit=time.time()+30
 while True:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
  except BlockingIOError:
   assert time.time()<limit,'old lock not released';time.sleep(1)
 write(OPS/'old_controller_retired.json',dict(old_pid=expected['old_pid'],only_after_child_zombie=True,training_audit_passed=True,original_lock_reacquired=True,time=time.time()))
 return lock

def worker(key):
 gpu,slot=map(int,key.split(':'));manifest=read(OPS/'manifest.json');deadline=manifest['deadline_unix'];oldlock=None
 with (OPS/('slot_'+key.replace(':','_')+'.lock')).open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  state=dict(slot=key,gpu=gpu,gpu_uuid=UUID[gpu],worker_pid=os.getpid(),jobs=PLAN[key],completed=[],status='starting');target=OPS/('state_'+key.replace(':','_')+'.json');write(target,state)
  try:
   for job in PLAN[key]:
    adopted=job=='rmt_t__A0';state.update(job=job,status='adopting_train' if adopted else 'queued',child_pid=1189143 if adopted else None);write(target,state)
    if adopted:oldlock=adoption(state)
    for mode in phases(adopted):
     with (OPS/'admission.lock').open('a') as guard:
      fcntl.flock(guard,fcntl.LOCK_EX)
      assert allowed((OPS/'blocked.json').exists(),time.time(),deadline),'new phase blocked or deadline expired'
      verify();assert not (RUN/job/('preflight' if mode=='preflight' else 'run' if mode=='train' else 'final_test')).exists(),'refuse duplicate phase '+job+' '+mode
      env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=UUID[gpu],EXPERIMENT_PHYSICAL_GPU=str(gpu),EXPERIMENT_GPU_UUID=UUID[gpu],OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
      log=(OPS/(job+'_'+mode+'.log')).open('x')
      print('START',key,job,mode,flush=True)
      child=subprocess.Popen([PY,'-B','-u',str(OLD/'entry.py'),mode,job,str(gpu)],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
      state.update(status=mode,child_pid=child.pid,phase_started_unix=time.time());write(target,state)
     # Parent remains the pane foreground process and keeps the read end alive.
     for line in child.stdout:print(line,end='',flush=True);log.write(line);log.flush()
     rc=child.wait();log.close();write(OPS/(job+'_'+mode+'_exitcode.json'),dict(exitcode=rc,pid=child.pid,time=time.time()));assert rc==0,job+' '+mode+' failed'
     if mode=='train':train_audit(job)
    c=read(RUN/job/'completion.json');assert c['complete'] and c['epochs']==300 and c['test_images_per_mode']==2020
    for mode in ['none','mirror4']:assert read(RUN/job/'final_test'/mode/'artifact_audit.json')['complete']
    state['completed'].append(job);write(target,state)
   state.update(status='complete',child_pid=None);write(target,state)
  except BaseException as e:
   block(key,e);state.update(status='blocked',error=repr(e));write(target,state);raise

def setup():
 fixture();p=verify();assert not (OPS/'manifest.json').exists()
 oldpid=1187819;childpid=1189143
 assert b'entry.py\x00controller' in Path('/proc/%d/cmdline'%oldpid).read_bytes()
 before=proc(childpid);assert before['ppid']==oldpid and before['state']!='Z'
 # tmux 3.4 automatically SIGCONTs a SIGSTOPped direct pane child. Use a
 # dedicated user-systemd scope containing ONLY the controller, not its child.
 scope='encoder-swap-old-controller-only.scope'
 subprocess.run(['busctl','--user','call','org.freedesktop.systemd1','/org/freedesktop/systemd1','org.freedesktop.systemd1.Manager','StartTransientUnit','ssa(sv)a(sa(sv))',scope,'fail','2','Description','s','EncoderSwap old scheduler only; preserve training','PIDs','au','1',str(oldpid),'0'],check=True)
 cg=subprocess.check_output(['systemctl','--user','show',scope,'-p','ControlGroup','--value'],text=True).strip();base=Path('/sys/fs/cgroup'+cg)
 assert set(map(int,(base/'cgroup.procs').read_text().split()))=={oldpid}
 assert cg not in Path('/proc/%d/cgroup'%childpid).read_text()
 subprocess.run(['systemctl','--user','freeze',scope],check=True)
 limit=time.time()+10
 while 'frozen 1' not in (base/'cgroup.events').read_text():assert time.time()<limit;time.sleep(.1)
 s=read(RUN/'controller_status.json');assert set(s['assigned'])=={'rmt_t__A0'}
 assert Path('/proc/%d/fd/1'%childpid).resolve()==RUN/'rmt_t__A0/train.log'
 write(OPS/'takeover.json',dict(old_pid=oldpid,old_state=proc(oldpid),child_pid=childpid,child_start_ticks=before['start_ticks'],child_state=proc(childpid),scope=scope,freezer_events=str(base/'cgroup.events'),scope_members=(base/'cgroup.procs').read_text(),stdout=str(Path('/proc/%d/fd/1'%childpid).resolve()),policy='Dedicated user-systemd cgroup freezer controller only; train stays in original scope untouched. SIGSTOP rejected: tmux auto-SIGCONT verified. After zombie exit0+training audit, SIGKILL controller only and acquire original lock. Never thaw old dispatch.',time=time.time()))
 write(OPS/'manifest.json',dict(plan=PLAN,slots_per_physical_gpu=3,existing_other_groups_not_counted=True,adopted_job='rmt_t__A0',deadline_unix=time.time()+336*3600,protocol_sha256=sha(RUN/'protocol.json'),entry=str(OLD/'entry.py'),cpu_acceptance_reused_count=12,scientific_changes=False,fail_policy='Any failure blocks subsequent phases/arms, never terminates running children',authorization='User explicitly supersedes batch reservations only; no other queue modified'))
 for key,jobs in PLAN.items():write(OPS/('state_'+key.replace(':','_')+'.json'),dict(status='queued',slot=key,jobs=jobs))
 print('SETUP_READY_OLD_CONTROLLER_STOPPED',flush=True)

def status():
 print('TIME',time.strftime('%Y-%m-%d %H:%M:%S %z'));print((OPS/'manifest.json').read_text())
 for f in sorted(OPS.glob('state_*.json')):print(f.name,f.read_text())
 print('OLD',proc(1187819) if Path('/proc/1187819').exists() else 'retired')
 print(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True))
 print(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True))
 for key,jobs in PLAN.items():
  for job in jobs:
   f=RUN/job/'first_optimizer_update.json';print('FIRST_STEP',job,read(f) if f.exists() else 'not yet')
   for name in ['preflight/acceptance.json','run/artifact_audit.json','completion.json']:
    f=RUN/job/name
    if f.exists():print('ARTIFACT',str(f),f.stat().st_size,f.stat().st_mtime)
   logs=sorted(OPS.glob(job+'*.log'))
   for f in logs[-1:]:print('LOG',str(f),f.stat().st_mtime,'\n',''.join(f.open().readlines()[-4:]))
if __name__=='__main__':
 if sys.argv[1]=='setup':setup()
 elif sys.argv[1]=='fixture':fixture()
 elif sys.argv[1]=='status':status()
 else:worker(sys.argv[2])

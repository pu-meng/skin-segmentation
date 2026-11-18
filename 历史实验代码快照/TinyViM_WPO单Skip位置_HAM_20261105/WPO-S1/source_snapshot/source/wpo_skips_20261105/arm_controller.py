import os,sys,subprocess,json,time,fcntl,threading
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from source.wpo_skips_20261105.selection import ARM,RUN,QUEUE
UUID='GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde'
os.environ.update(CUDA_VISIBLE_DEVICES=UUID,EXPERIMENT_PHYSICAL_GPU='0',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',MKL_THREADING_LAYER='GNU')
RUN.mkdir(parents=True,exist_ok=True)
def write(name,data):
 p=RUN/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def snapshot(child):
 out=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
 rows=[r.strip().split(', ') for r in out.splitlines() if ', ' in r]
 matches=[r for r in rows if int(r[1])==child.pid]
 if matches:assert all(r[0]==UUID for r in matches),'wrong GPU UUID'
 return dict(checked_pid=child.pid,gpu_uuid=UUID,matches=matches,all_compute=out)
with (RUN/'controller.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state=dict(arm=ARM,controller_pid=os.getpid(),gpu=0,gpu_uuid=UUID,shared_authorized=True)
 start=time.monotonic()
 try:
  for phase in ['cpu','preflight','train','test']:
   if phase=='cpu' and (RUN/'cpu_check.json').exists():
    proof=json.loads((RUN/'cpu_check.json').read_text());assert proof['passed'] and proof['source_sha256']=={str(p.name):__import__('hashlib').sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    print('REUSE_PASSED_CPU',ARM,flush=True);continue
   if phase=='preflight':
    state.update(status='waiting_pair_prepare');write('controller_status.json',state)
    pair=['WPO-S0','WPO-S1'] if ARM in ['WPO-S0','WPO-S1'] else ['WPO-S2']
    deadline=time.monotonic()+1800
    while True:
     for a in pair:
      p=QUEUE/a/'controller_status.json'
      if p.exists():assert json.loads(p.read_text()).get('status')!='failed','paired preparation failed'
     if all((QUEUE/a/'prepare.exitcode').exists() and (QUEUE/a/'prepare.exitcode').read_text().strip()=='0' for a in pair):break
     assert time.monotonic()<deadline,'paired prepare timeout';time.sleep(2)
   if phase=='train':
    state.update(status='waiting_pair_preflight');write('controller_status.json',state)
    pair=['WPO-S0','WPO-S1'] if ARM in ['WPO-S0','WPO-S1'] else ['WPO-S2']
    deadline=time.monotonic()+1800
    while True:
     for a in pair:
      p=QUEUE/a/'controller_status.json'
      if p.exists():assert json.loads(p.read_text()).get('status')!='failed','paired arm failed; train blocked'
     if all((QUEUE/a/'preflight.exitcode').exists() and (QUEUE/a/'preflight.exitcode').read_text().strip()=='0' for a in pair):break
     assert time.monotonic()<deadline,'paired preflight timeout';time.sleep(5)
   if phase=='cpu':args=['-m','source.wpo_skips_20261105.check']
   else:args=['-m','source.wpo_skips_20261105.entry',phase]
   env=os.environ.copy()
   if phase=='cpu':env['CUDA_VISIBLE_DEVICES']=''
   with (RUN/(phase+'_console.log')).open('x') as f:
    child=subprocess.Popen([sys.executable,'-B','-u']+args,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    state.update(status=phase,child_pid=child.pid,updated_at=datetime.now().astimezone().isoformat());write('controller_status.json',state)
    def sample():
     while child.poll() is None:
      try:write(phase+'_gpu_snapshot.json',snapshot(child))
      except BaseException as e:print('GPU sample warning',repr(e),flush=True)
      time.sleep(3)
    thread=threading.Thread(target=sample,daemon=True);thread.start()
    timer=threading.Timer(max(1,120*3600-(time.monotonic()-start)),child.terminate);timer.daemon=True;timer.start()
    print(ARM,phase,'PID',child.pid,flush=True)
    for line in child.stdout:print(line,end='',flush=True);f.write(line);f.flush()
    rc=child.wait();timer.cancel();(RUN/(phase+'.exitcode')).write_text(str(rc)+'\n');assert rc==0,f'{phase} failed {rc}; no retry'
   if phase=='train':
    a=json.loads((RUN/'run/artifact_audit.json').read_text());assert a['training_complete'] and a['epochs']==300 and a['validation_artifacts_complete']
   if phase=='test':
    a=json.loads((RUN/'final_test'/ARM/'artifact_audit.json').read_text());assert a['complete'] and a['prediction_count']==2020
  state.update(status='complete',updated_at=datetime.now().astimezone().isoformat());write('controller_status.json',state);(RUN/'exitcode').write_text('0\n')
 except BaseException as e:
  state.update(status='failed',error=repr(e),updated_at=datetime.now().astimezone().isoformat());write('controller_status.json',state);(RUN/'exitcode').write_text('1\n');raise


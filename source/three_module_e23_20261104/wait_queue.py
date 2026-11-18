"""Bounded per-arm detached queue. Never share GPUs or preempt prior reservations."""
import os,sys,json,time,subprocess,fcntl,hashlib,signal
from pathlib import Path
from datetime import datetime
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PY='/home/PuMengYu/anaconda3/envs/medseg/bin/python'
ARM=sys.argv[1];GPU=1
RUN=ROOT/('log/TinyViM_三模块_'+ARM+('_HSA' if ARM=='E2' else '_FDCG')+'_HAM_20261104')
GUARD=json.loads((HERE/'reservations.json').read_text())[str(GPU)]
DEADLINE=time.time()+96*3600
state=dict(arm=ARM,gpu=GPU,controller_pid=os.getpid(),deadline=datetime.fromtimestamp(DEADLINE).astimezone().isoformat())
def save(status,**kw):
 state.update(status=status,updated_at=datetime.now().astimezone().isoformat(),**kw)
 p=RUN/'controller_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2));tmp.replace(p)
 print(json.dumps(state,ensure_ascii=False),flush=True)
def identity(pid):
 try:
  s=Path(f'/proc/{pid}/stat').read_text().split();return s[21] if s[2]!='Z' else None
 except FileNotFoundError:return None
def resources():
 active=[r['pid'] for r in GUARD if identity(r['pid'])==r['start']]
 # UUID/index query is performed against this physical card only.
 busy=subprocess.check_output(['nvidia-smi','-i',str(GPU),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
 return active,busy
ENV=os.environ.copy();ENV.update(E23_ARM=ARM,CUDA_VISIBLE_DEVICES=str(GPU),EXPERIMENT_PHYSICAL_GPU=str(GPU),PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
def phase(name,args,seconds=72*3600):
 logfile=RUN/(name+'_console.log')
 with logfile.open('x') as stream:
  child=subprocess.Popen([PY,'-u',*args],cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
  save(name,child_pid=child.pid,log=str(logfile))
  try:rc=child.wait(timeout=min(seconds,max(1,DEADLINE-time.time())))
  except subprocess.TimeoutExpired:
   # Only this controller's newly created worker group, never prior jobs.
   os.killpg(child.pid,signal.SIGTERM);save('failed',error=name+' deadline');raise
  (RUN/(name+'.exitcode')).write_text(str(rc))
  if rc:raise RuntimeError(name+' exit '+str(rc))

def main():
 RUN.mkdir(exist_ok=True)
 with (RUN/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert not (RUN/'controller_status.json').exists(),'No implicit retry/overwrite'
  (RUN/'controller.pid').write_text(str(os.getpid()))
  try:
   # E3 CPU acceptance already launched externally. Receive its actual PASS JSON;
   # do not restart it, rely on successful terminal JSON plus process completion.
   cp=HERE/('cpu_'+ARM+'_acceptance.json')
   while not cp.exists():
    pid=2036058 if ARM=='E3' else -1
    if pid<0 or identity(pid) is None:raise RuntimeError('CPU acceptance exited without PASS JSON')
    if time.time()>DEADLINE:raise TimeoutError('CPU wait deadline')
    save('waiting_cpu_acceptance',upstream_cpu_pid=pid,cpu_log=str(HERE/'cpu_E3_attempt1.log'));time.sleep(30)
   proof=json.loads(cp.read_text());assert proof['passed'] and not proof['cuda_initialized'] and proof['scan_restored']
   if ARM=='E3':
    upstream=ROOT/'log/TinyViM_三模块_E2_HSA_HAM_20261104'
    while True:
     if time.time()>DEADLINE:raise TimeoutError('E2 handoff deadline')
     sp=upstream/'controller_status.json'
     us=json.loads(sp.read_text()) if sp.exists() else {}
     if us.get('status')=='failed':raise RuntimeError('E2 failed; E3 blocked without automatic retry')
     if us.get('status')=='complete' and (upstream/'controller.exitcode').read_text().strip()=='0':
      ua=json.loads((upstream/'final_test/E2/artifact_audit.json').read_text())
      assert ua['complete'] and ua['prediction_count']==2020 and ua['metric_count']==2020 and ua['visualizations']==12
      if identity(us['controller_pid']) is None:break
     save('waiting_E2_complete',upstream_status=us.get('status'),upstream_controller_pid=us.get('controller_pid'));time.sleep(30)
   while True:
    if time.time()>DEADLINE:raise TimeoutError('GPU wait deadline')
    active,busy=resources()
    if not active and not busy:break
    save('waiting_gpu_and_prior_pipeline',reserved_controller_pids=active,compute_pids=busy);time.sleep(60)
   # Hold per-GPU advisory lock through all own phases (no gaps between train/test).
   with (HERE/('gpu'+str(GPU)+'.lock')).open('a') as glock:
    fcntl.flock(glock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    active,busy=resources();assert not active and not busy,'GPU became busy'
    phase('gpu_acceptance',[str(HERE/'acceptance.py'),'gpu',ARM],3600)
    assert json.loads((HERE/('gpu_'+ARM+'_acceptance.json')).read_text())['passed']
    phase('prepare',[str(HERE/'entry.py'),'prepare'],600)
    phase('preflight',[str(HERE/'entry.py'),'preflight'],3600)
    phase('train',[str(HERE/'entry.py'),'train'])
    phase('test',[str(HERE/'entry.py'),'test'],12*3600)
    audit=json.loads((RUN/'final_test'/ARM/'artifact_audit.json').read_text());assert audit['complete'] and audit['prediction_count']==2020 and audit['metric_count']==2020 and audit['visualizations']==12
    save('complete',test_artifact_audit=audit);(RUN/'controller.exitcode').write_text('0')
  except BaseException as e:
   save('failed',error=repr(e));(RUN/'controller.exitcode').write_text('1');raise
if __name__=='__main__':main()

"""Single-instance joint E2.2: CPU -> CUDA -> real batch8 -> HAM300 -> fixed-best test."""
import os,sys,subprocess,json,time,fcntl,hashlib
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).parent;ROOT=HERE.parents[1];RUN=ROOT/'log/TinyViM_E2.2_新版全局局部_HAM_20261105'
UUID='GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b'
os.environ.update(CUDA_VISIBLE_DEVICES=UUID,EXPERIMENT_PHYSICAL_GPU='1',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',MKL_THREADING_LAYER='GNU')
RUN.mkdir(exist_ok=True)
def write(name,data):
 p=RUN/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def smi():
 return dict(gpu=subprocess.check_output(['nvidia-smi','-i',UUID,'--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip(),compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True).strip())
with (RUN/'controller.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state=dict(controller_pid=os.getpid(),gpu=1,gpu_uuid=UUID,shared_authorized=True,scope='Standalone E2.2 authorized in addition to continuing E123.1; E1/E3 disabled',deadline_hours=96)
 start=time.monotonic()
 try:
  write('gpu_initial.json',smi())
  for phase,args in [('cpu',[str(HERE/'check.py'),'cpu']),('gpu',[str(HERE/'check.py'),'gpu']),('prepare',[str(HERE/'entry.py'),'prepare']),('preflight',[str(HERE/'entry.py'),'preflight']),('train',[str(HERE/'entry.py'),'train']),('test',[str(HERE/'entry.py'),'test'])]:
   if phase in ['cpu','gpu'] and (HERE/(phase+'_check.json')).exists():
    proof=json.loads((HERE/(phase+'_check.json')).read_text());assert proof['passed'];print('REUSE_PASSED_ACCEPTANCE',phase,flush=True);continue
   if phase in ['gpu','preflight']:
    free=int(subprocess.check_output(['nvidia-smi','-i',UUID,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>=6000,f'GPU1 remaining {free} MiB < required6000; no batch/GPU change'
   log=RUN/(phase+'_console.log')
   env=os.environ.copy()
   if phase=='cpu':env['CUDA_VISIBLE_DEVICES']=''
   with log.open('x') as f:
    child=subprocess.Popen([sys.executable,'-B','-u']+args,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=env)
    state.update(status=phase,child_pid=child.pid,log=str(log),updated_at=datetime.now().astimezone().isoformat());write('controller_status.json',state);print(json.dumps(state),flush=True)
    # deadline per phase; stdout remains visible in tmux and tee phase log.
    import threading
    def terminate_own():
     state.update(deadline_exceeded=True);write('controller_status.json',state)
     child.terminate()
    timer=threading.Timer(max(1,96*3600-(time.monotonic()-start)),terminate_own);timer.daemon=True;timer.start()
    for line in child.stdout:print(line,end='',flush=True);f.write(line);f.flush()
    rc=child.wait();timer.cancel();(RUN/(phase+'.exitcode')).write_text(str(rc)+'\n');assert rc==0,f'{phase} failed {rc}; no retry'
   write(phase+'_gpu_snapshot.json',smi())
   if phase=='train':
    audit=json.loads((RUN/'run/artifact_audit.json').read_text());assert audit['training_complete'] and audit['epochs']==300
  audit=json.loads((RUN/'final_test/E2.2/artifact_audit.json').read_text());assert audit['complete'] and audit['prediction_count']==2020
  state.update(status='complete',updated_at=datetime.now().astimezone().isoformat());write('controller_status.json',state)
 except BaseException as e:
  state.update(status='failed',error=repr(e),updated_at=datetime.now().astimezone().isoformat());write('controller_status.json',state);raise


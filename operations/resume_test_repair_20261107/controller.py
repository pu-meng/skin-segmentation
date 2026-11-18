import os,sys,json,subprocess,fcntl,time,hashlib
from pathlib import Path
OPS=Path(__file__).resolve().parent
ROOT=OPS.parents[1]
ARMS=['R0','R1','R2','R3','DualA']
def write(v):(OPS/'queue_status.json').write_text(json.dumps(v,ensure_ascii=False,indent=2))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 with (OPS/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  state={'pid':os.getpid(),'gpu':0,'arms':ARMS,'completed':[],'pending':ARMS.copy(),'status':'running'};write(state)
  for arm in ARMS:
   state.update(current=arm,pending=ARMS[ARMS.index(arm)+1:]);write(state)
   base=Path('/media/8T/PuMengYu/experiments/skin_segmentation')/('DecoderDualA_RESUME100to300_20261106' if arm=='DualA' else 'TinyViM_E1232_FourArm_RESUME100to300_20261106')
   protected=[base/arm/'train'/n for n in ['history.csv','checkpoints/best.pth','checkpoints/latest.pth']];before={str(p):sha(p) for p in protected}
   env=os.environ.copy();env.update(RESUME_ARM=arm,CUDA_VISIBLE_DEVICES='GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde',EXPERIMENT_PHYSICAL_GPU='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
   print('START_FINAL_TEST',arm,flush=True)
   with (OPS/(arm+'.log')).open('x') as log:
    child=subprocess.Popen([sys.executable,'-B','-u',str(OPS/'entry.py'),'test'],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    state['child_pid']=child.pid;write(state)
    for line in child.stdout:print(line,end='',flush=True);log.write(line);log.flush()
    code=child.wait()
   state['last_exitcode']=code
   if code!=0:state.update(status='failed');write(state);return 1
   audit=json.loads((base/'final_test'/arm/'artifact_audit.json').read_text());assert audit['complete'] and audit['prediction_count']==2020
   assert all(sha(p)==before[str(p)] for p in protected),'protected training files changed'
   state['completed'].append(arm);write(state)
  state.update(status='complete',current=None,pending=[]);write(state);return 0
if __name__=='__main__':sys.exit(main())

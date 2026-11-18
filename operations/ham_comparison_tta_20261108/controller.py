import os,sys,subprocess,json,time,fcntl,threading,queue,traceback
from pathlib import Path
OPS=Path(__file__).resolve().parent;ROOT=OPS.parents[1];OUT=Path('/media/8T/PuMengYu/experiments/skin_segmentation/HAM_comparison_TTA_20261108');PY='/home/PuMengYu/anaconda3/envs/medseg/bin/python'
sys.path.insert(0,str(OPS));from publish import publish
m=json.loads((OPS/'manifest.json').read_text());pending=queue.Queue()
for e in m['rows']:pending.put(e)
mutex=threading.Lock();state=dict(pid=os.getpid(),status='running',completed=[],failed=[],slots={},dataset='HAM10000',test_images=2020,expected_new=16)
def save():
 state['updated_unix']=time.time();tmp=OPS/'queue_status.tmp';tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2));tmp.replace(OPS/'queue_status.json')
def worker(gpu):
 while True:
  try:e=pending.get_nowait()
  except queue.Empty:return
  ident=e['id'];env=os.environ.copy();env.update(TTA_PHYSICAL_GPU=str(gpu),CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',MKL_THREADING_LAYER='GNU',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
  with (OUT/ident/'console.log').open('x') as log:
   p=subprocess.Popen([PY,'-B','-u',str(OPS/'worker.py'),ident],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
   with mutex:state['slots'][str(gpu)]=dict(id=ident,pid=p.pid,status='testing');save()
   rc=p.wait()
  with mutex:
   state['completed' if rc==0 else 'failed'].append(ident);state['slots'][str(gpu)].update(status='complete' if rc==0 else 'failed',exitcode=rc);save();publish()
with (OPS/'controller.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);save();publish()
 ts=[threading.Thread(target=worker,args=(g,)) for g in [1,0]]
 for t in ts:t.start()
 for t in ts:t.join()
 state['status']='complete' if not state['failed'] else 'finished_with_failures';save();publish()

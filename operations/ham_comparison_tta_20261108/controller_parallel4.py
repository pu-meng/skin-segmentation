"""Four TTA-only slots; adopt existing inference workers without interruption."""
import os,sys,json,time,subprocess,threading,queue,fcntl,traceback
from pathlib import Path
OPS=Path(__file__).resolve().parent;ROOT=OPS.parents[1];OUT=Path('/media/8T/PuMengYu/experiments/skin_segmentation/HAM_comparison_TTA_20261108');PY='/home/PuMengYu/anaconda3/envs/medseg/bin/python'
sys.path.insert(0,str(OPS));from publish import publish
manifest=json.loads((OPS/'manifest.json').read_text());take=json.loads((OPS/'parallel4_takeover.json').read_text());adopt={int(g):v for g,v in take['adopted'].items()};adopt_ids={v['id'] for v in adopt.values()}
pending=queue.Queue()
for e in manifest['rows']:
 if e['id'] not in adopt_ids and e['id'] not in take['completed'] and e['id'] not in take['failed']:pending.put(e['id'])
mutex=threading.Lock();state=dict(pid=os.getpid(),status='running',parallel_slots=4,slots_per_gpu=2,scope='HAM TTA补测16项 only; no training or other queues modified',completed=list(take['completed']),failed=list(take['failed']),slots={},dataset='HAM10000',test_images=2020,expected_new=16)
def read(p):return json.loads(Path(p).read_text())
def save():
 state['updated_unix']=time.time();tmp=OPS/'queue_status.tmp';tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2));tmp.replace(OPS/'queue_status.json')
def proc(pid):
 try:
  t=Path('/proc/%d/stat'%pid).read_text();v=t[t.rindex(')')+2:].split();return v[0],int(v[19])
 except FileNotFoundError:return None

def wait_adopt(v):
 end=time.time()+4*3600
 while time.time()<end:
  p=OUT/v['id'];a=p/'artifact_audit.json';s=p/'status.json';live=proc(v['pid'])
  if a.exists() and read(a).get('complete') and s.exists() and read(s).get('status')=='complete':return 0
  if live is None or live[0]=='Z' or live[1]!=v['start_ticks'] or (p/'failure.json').exists():return 1
  time.sleep(5)
 return 1

def slot(gpu,index):
 key=f'{gpu}:{index}';initial=adopt.get(gpu) if index==0 else None
 while True:
  if initial:ident=initial['id']
  else:
   try:ident=pending.get_nowait()
   except queue.Empty:return
  try:
   if initial:
    with mutex:state['slots'][key]=dict(id=ident,pid=initial['pid'],status='adopting_existing_inference');save()
    rc=wait_adopt(initial);initial=None
   else:
    # Small batch1 inference shares the existing load; no idle GPU condition.
    assert __import__('shutil').disk_usage(OUT).free>8*2**30,'disk reserve below8GiB'
    env=os.environ.copy();env.update(TTA_PHYSICAL_GPU=str(gpu),CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',MKL_THREADING_LAYER='GNU',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    with (OUT/ident/'console.log').open('x') as log:
     child=subprocess.Popen([PY,'-B','-u',str(OPS/'worker.py'),ident],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
     with mutex:state['slots'][key]=dict(id=ident,pid=child.pid,status='testing');save()
     rc=child.wait()
   if rc==0:assert read(OUT/ident/'artifact_audit.json')['complete']
   with mutex:
    state['completed' if rc==0 else 'failed'].append(ident);state['slots'][key].update(status='complete' if rc==0 else 'failed',exitcode=rc);save();publish()
  except BaseException as ex:
   with mutex:
    state['failed'].append(ident);state['slots'][key]=dict(id=ident,status='failed',error=repr(ex));save();publish()
   (OUT/ident/'controller_failure.json').write_text(json.dumps(dict(error=repr(ex),traceback=traceback.format_exc()),indent=2))
with (OPS/'controller.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);save();publish()
 threads=[threading.Thread(target=slot,args=(g,i)) for g in [1,0] for i in [0,1]]
 for t in threads:t.start()
 for t in threads:t.join()
 state['status']='complete' if not state['failed'] else 'finished_with_failures';save();publish()

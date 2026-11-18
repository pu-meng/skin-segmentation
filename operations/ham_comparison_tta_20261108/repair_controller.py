import os,sys,json,time,subprocess,threading,fcntl,traceback
from pathlib import Path
OPS=Path(__file__).resolve().parent;ROOT=OPS.parents[1];OUT=Path('/media/8T/PuMengYu/experiments/skin_segmentation/HAM_comparison_TTA_20261108');PY='/home/PuMengYu/anaconda3/envs/medseg/bin/python'
sys.path.insert(0,str(OPS));from publish import publish
state=dict(status='running',pid=os.getpid(),scope='RMT-T/RMT-S/SU-RMT failed import-only HAM TTA retry; no training',completed=[],failed=[],slots={});mutex=threading.Lock()
def read(p):return json.loads(Path(p).read_text())
def save():
 state['updated_unix']=time.time();p=OPS/'repair_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(state,ensure_ascii=False,indent=2));t.replace(p)
def active(pid):
 try:
  s=Path('/proc/%d/stat'%pid).read_text();return s[s.rindex(')')+2:].split()[0]!='Z'
 except FileNotFoundError:return False
def original_count(gpu):
 s=read(OPS/'queue_status.json');return sum(1 for k,v in s['slots'].items() if k.startswith(str(gpu)+':') and v.get('status') in ['testing','adopting_existing_inference'] and active(v.get('pid',0)))
def worker(ident,gpu):
 try:
  while True:
   with mutex:
    ours=sum(v.get('gpu')==gpu and v.get('status')=='testing' for v in state['slots'].values())
    if original_count(gpu)+ours<2:
     state['slots'][ident]=dict(gpu=gpu,status='testing');save();break
   time.sleep(2)
  env=os.environ.copy();env.update(TTA_PHYSICAL_GPU=str(gpu),CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',MKL_THREADING_LAYER='GNU',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
  with (OUT/ident/'console.log').open('x') as log:
   p=subprocess.Popen([PY,'-B','-u',str(OPS/'worker.py'),ident],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
   with mutex:state['slots'][ident]['pid']=p.pid;save()
   rc=p.wait()
  with mutex:
   state['completed' if rc==0 else 'failed'].append(ident);state['slots'][ident].update(status='complete' if rc==0 else 'failed',exitcode=rc);save();publish()
 except BaseException as e:
  with mutex:state['failed'].append(ident);state['slots'][ident]=dict(gpu=gpu,status='failed',error=repr(e));save();publish()
with (OPS/'repair_controller.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert read(OPS/'repair_cpu_acceptance.json')['passed'];old=read(OPS/'queue_status.json')
 assert len(old['completed'])+len(old['failed'])+sum(v.get('status') in ['testing','adopting_existing_inference'] for v in old['slots'].values())==16,'original dispatcher still has unassigned work'
 for ident in ['rmt_t','rmt_s','su_rmt']:
  p=OUT/ident;assert (p/'failure.json').exists() and not (p/'mirror4').exists();archive=OUT/'failed_import_attempt1'/ident;archive.parent.mkdir(exist_ok=True);p.rename(archive);p.mkdir();__import__('shutil').copy2(archive/'test_ids.json',p/'test_ids.json');state['slots'][ident]=dict(status='queued')
 save();publish();threads=[threading.Thread(target=worker,args=(i,g)) for i,g in [('rmt_t',0),('rmt_s',1),('su_rmt',1)]]
 for t in threads:t.start()
 for t in threads:t.join()
 state['status']='complete' if not state['failed'] else 'finished_with_failures';save();publish()
 # Preserve the main dispatch state until its own pending workers have finished.
 while active(old['pid']):time.sleep(2)
 m=read(OPS/'queue_status.json');ids=[e['id'] for e in read(OPS/'manifest.json')['rows']];m['completed']=[i for i in ids if (OUT/i/'artifact_audit.json').exists() and read(OUT/i/'artifact_audit.json').get('complete')];m['failed']=[i for i in ids if i not in m['completed']];m.update(status='complete' if not m['failed'] else 'finished_with_failures',repair_controller_pid=os.getpid(),updated_unix=time.time());(OPS/'queue_status.json').write_text(json.dumps(m,ensure_ascii=False,indent=2));publish()

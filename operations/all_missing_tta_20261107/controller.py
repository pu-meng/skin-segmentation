import os,sys,json,time,subprocess,fcntl,hashlib,collections,shutil
from pathlib import Path
R=Path('/home/PuMengYu/experiments/skin_segmentation');OPS=R/'operations/all_missing_tta_20261107';OUT=Path('/media/8T/PuMengYu/experiments/skin_segmentation/all_missing_tta_20261107')
def save(p,v):
 p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def load(p):return json.loads(Path(p).read_text())
def publish(m):
 doc=Path(m['document']);raw=doc.read_bytes();text=raw.decode();changes=[]
 for e in m['rows']:
  if e['status']!='complete':continue
  a=load(OUT/e['id']/'artifact_audit.json');s=load(Path(e['output'])/'summary.json');assert a['complete'] and a['protected_unchanged'] and s['checkpoint_sha256']==e['checkpoint_sha256'] and s['split_sha256']==e['split_sha256']
  old=e['original_row'];new=old.replace('|未取得|',f"|{100*s['global']['dice']:.4f}|")
  if new in text.splitlines():continue
  assert text.splitlines().count(old)==1,'unique row matching failed '+e['id'];text=text.replace(old,new,1);changes.append({'id':e['id'],'old':old,'new':new,'summary':e['output']+'/summary.json'})
 if not changes:return
 # Patch only exact original four-column rows against freshly read bytes.
 # Keep all other content and existing metric cells byte-for-byte unchanged.
 assert doc.read_bytes()==raw,'concurrent document edit; retry publish later'
 backup=OPS/('document_before_'+str(time.time_ns())+'.md');backup.write_bytes(raw)
 tmp=doc.with_name(doc.name+'.tta_'+str(os.getpid())+'.tmp');tmp.write_bytes(text.encode());assert doc.read_bytes()==raw,'concurrent document edit';tmp.replace(doc);assert doc.read_bytes()==text.encode()
 save(OPS/('publish_'+str(time.time_ns())+'.json'),dict(changes=changes,before_sha256=hashlib.sha256(raw).hexdigest(),after_sha256=hashlib.sha256(doc.read_bytes()).hexdigest(),readback_verified=True))
def main():
 with (OPS/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);m=load(OPS/'manifest.json');state=dict(pid=os.getpid(),gpu=1,gpu_uuid=m['gpu_uuid'],status='running',deadline_unix=m['deadline_unix']);(OPS/'controller.pid').write_text(str(os.getpid()))
  def update():
   state['counts']=dict(collections.Counter(e['status'] for e in m['rows']));state['updated_unix']=time.time();save(OPS/'manifest.json',m);save(OUT/'manifest.json',m);save(OPS/'queue_status.json',state)
  update()
  for e in m['rows']:
   if e['status'] in ['complete','blocked']:continue
   state['current']=e['id'];state['name']=e['name'];update()
   while True:
    if time.time()>m['deadline_unix']:
     e.update(status='blocked',blocker='72h total deadline');update();break
    free=int(subprocess.check_output(['nvidia-smi','-i',m['gpu_uuid'],'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    disk=shutil.disk_usage(OUT).free;state['free_memory_MiB']=free;state['disk_free_bytes']=disk
    if free>=4000 and disk>=8*2**30:break
    e['status']='waiting_resources';state['status']='waiting_resources';update();print('WAIT_GPU1_MEMORY',free,flush=True);time.sleep(60)
   if e['status']=='blocked':continue
   e['status']='running';state['status']='running';update()
   work=OUT/e['id'];log=work/'console.log';env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_THREADING_LAYER='GNU')
   with log.open('x') as f:
    p=subprocess.Popen([e['python'],'-B','-u',str(OPS/'worker.py'),e['id']],cwd=R,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=env);e['pid']=p.pid;state['child_pid']=p.pid;update();print('LAUNCH',e['id'],e['name'],p.pid,flush=True)
    for line in p.stdout:print(line,end='',flush=True);f.write(line);f.flush()
    code=p.wait()
   (work/'exitcode').write_text(str(code)+'\n');e['exitcode']=code
   if code==0 and load(work/'artifact_audit.json').get('complete'):
    e['status']='complete';e['result']=load(Path(e['output'])/'summary.json');e['blocker']=None
   else:e['status']='blocked';e['blocker']=load(work/'failure.json') if (work/'failure.json').exists() else {'error':'worker exit '+str(code)}
   update()
   try:publish(m);state.pop('publish_error',None)
   except Exception as ex:state['publish_error']=repr(ex);update();print('PUBLISH_BLOCKED',repr(ex),flush=True)
  state.update(status='finished_with_blocks' if any(e['status']=='blocked' for e in m['rows']) else 'complete',current=None,child_pid=None);update();publish(m);save(OUT/'results.json',dict(complete=state['status']=='complete',counts=state['counts'],rows=m['rows']));(OPS/'exitcode').write_text('0\n');print('QUEUE_FINISHED',state['counts'],flush=True)
if __name__=='__main__':main()

import os,sys,time,json,fcntl,subprocess,threading,signal,traceback
from pathlib import Path
R=Path('/home/PuMengYu/experiments/skin_segmentation')
O=R/'operations/gpu1_takeover_20261112'
A=Path('/media/8T/PuMengYu/experiments/skin_segmentation/ABLATION_REPAIR_FRESH300_20261112')
E=Path('/media/8T/PuMengYu/experiments/skin_segmentation/EXPLORATION_REPAIR_FRESH300_20261112')
PY='/home/PuMengYu/anaconda3/envs/medseg/bin/python'
GPU='GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b'
entries={'a':R/'operations/ablation_repair_fresh_20261112/runner.py','e':R/'operations/exploration_protocol_repair_20261112/runner.py'}
arms=[('a','swift','no_e2'),('a','swift','no_d2d3'),('e','keepS3','ISIC2018'),('e','convnextS','ISIC2018'),('e','D2Mskips','ISIC2018'),('e','MSDdec','ISIC2018'),('e','swiftS2S3','HAM10000'),('a','a1s1','no_skip1'),('a','a1s1','no_e2'),('a','a1s1','no_d2d3')]
old=[('a','swift','no_skip1',844648,844930),('e','keepS2','ISIC2018',849013,849290)]
mutex=threading.Lock(); deadline=time.time()+30*86400

def name(a):return a[0]+'__'+a[1]+'__'+a[2]
def work(a):return (A/(a[1]+'__'+a[2])) if a[0]=='a' else (E/a[1])
def write(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(d,indent=2,default=str)+'\n');os.replace(t,p)
def state(a,phase,**kw):
 with mutex: write(O/'states'/(name(a)+'.json'),dict(arm=a,phase=phase,time=time.time(),**kw))
def run(a,phase):
 w=work(a);w.mkdir(parents=True,exist_ok=True)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='' if phase=='cpu' else GPU,EXPERIMENT_PHYSICAL_GPU='1',EXPERIMENT_GPU_UUID=GPU,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 log=O/'logs'/(name(a)+'__'+phase+'.log');log.parent.mkdir(parents=True,exist_ok=True)
 if log.exists():raise RuntimeError('refusing existing log '+str(log))
 if phase=='train' and (w/'run').exists():raise RuntimeError('refusing existing run '+str(w))
 if phase=='test' and ((w/'test').exists() or (w/'final_test').exists()):raise RuntimeError('refusing existing test '+str(w))
 with log.open('x') as f:
  p=subprocess.Popen([PY,'-B','-u',str(entries[a[0]]),'phase',a[1],a[2],phase],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  state(a,phase,pid=p.pid,log=str(log));rc=p.wait()
 write(O/'exits'/(name(a)+'__'+phase+'.json'),{'pid':p.pid,'exitcode':rc,'log':str(log),'time':time.time()})
 if rc:raise RuntimeError(phase+' failed rc='+str(rc)+' '+str(log))
 if phase=='preflight':
  d=json.loads((w/'preflight.json').read_text());assert d['passed'] and d['batch_shape']==[8,3,256,256]
 if phase=='test':
  p=w/('test_artifact_audit.json' if a[0]=='a' else 'completion.json');assert json.loads(p.read_text()).get('complete')
def go(a):
 try:
  with (O/'locks'/(name(a)+'.lock')).open('a') as l:
   fcntl.flock(l,fcntl.LOCK_EX|fcntl.LOCK_NB)
   if a[0]=='e' and not (work(a)/'cpu_acceptance.json').exists():run(a,'cpu')
   run(a,'preflight');run(a,'train');run(a,'test');state(a,'complete')
 except BaseException as exc:state(a,'failed',error=repr(exc),traceback=traceback.format_exc())
def oldgo(x):
 a=x[:3];parent,child=x[3:]
 try:
  state(a,'existing_training',pid=child,parent=parent)
  while time.time()<deadline:
   stat=Path('/proc/%d/stat'%child)
   if not stat.exists():raise RuntimeError('child disappeared before raw exit status obtained')
   data=stat.read_text();fields=data[data.rfind(')')+2:].split()
   if fields[0]=='Z':
    code=int(fields[49]);assert code==0,'raw wait status '+str(code)
    break
   time.sleep(10)
  else:raise RuntimeError('deadline')
  log=work(a)/'train.log';assert 'TRAIN_COMPLETE' in log.read_text(errors='replace')
  os.kill(parent,signal.SIGTERM)
  subprocess.run(['systemctl','--user','thaw','hermes-takeover-%d.scope'%parent],check=True)
  for _ in range(120):
   if not Path('/proc/%d'%parent).exists():break
   time.sleep(1)
  else:raise RuntimeError('parent not retired')
  with (work(a).parent/'controller.lock').open('a') as l:
   fcntl.flock(l,fcntl.LOCK_EX|fcntl.LOCK_NB)
  write(O/'exits'/(name(a)+'__original_train.json'),{'child':child,'raw_wait_status':code,'old_parent_retired':parent})
  with (O/'locks'/(name(a)+'.lock')).open('a') as l:
   fcntl.flock(l,fcntl.LOCK_EX|fcntl.LOCK_NB);run(a,'test');state(a,'complete')
 except BaseException as exc:state(a,'failed',error=repr(exc),traceback=traceback.format_exc())
def main():
 O.mkdir(parents=True,exist_ok=True)
 for p in ('locks','logs','states','exits'):(O/p).mkdir(exist_ok=True)
 with (O/'scheduler.lock').open('a') as l:
  fcntl.flock(l,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert not (O/'started.json').exists()
  write(O/'started.json',{'pid':os.getpid(),'gpu_uuid':GPU,'deadline':deadline,'pending':arms,'old':old})
  threads=[]
  for a in old:
   t=threading.Thread(target=oldgo,args=(a,));t.start();threads.append(t)
  pending=list(arms)
  while pending and time.time()<deadline:
   threads=[t for t in threads if t.is_alive()]
   while len(threads)<5 and pending:
    a=pending.pop(0);t=threading.Thread(target=go,args=(a,));t.start();threads.append(t)
   write(O/'queue.json',{'pending':pending,'active_slots':len(threads),'deadline':deadline,'time':time.time()})
   time.sleep(10)
  for t in threads:t.join()
  write(O/'queue.json',{'pending':pending,'active_slots':0,'deadline':deadline,'time':time.time(),'status':'finished' if not pending else 'deadline'})
if __name__=='__main__':main()

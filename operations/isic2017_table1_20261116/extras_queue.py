import fcntl,hashlib,json,os,subprocess,time,traceback
from pathlib import Path
R=Path('/home/PuMengYu/experiments/skin_segmentation');O=R/'operations/isic2017_table1_20261116';OUT=R/'log/ISIC2017_表1_去重_20261116';CFG=O/'extras_jobs.json';STATUS=O/'extras_status.json';PY='/home/PuMengYu/anaconda3/envs/medseg/bin/python';GPU='GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b'
states={};live={}
def load(p):return json.loads(p.read_text())
def save():
 obj={'pid':os.getpid(),'updated_at':time.time(),'jobs':{k:{'status':v,**({'pid':live[k][0].pid,'phase':live[k][1]} if k in live else {})} for k,v in states.items()}};tmp=STATUS.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(STATUS)
def free():
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader'],text=True).splitlines();return int(next(x.split(',')[1].strip().split()[0] for x in rows if x.startswith(GPU)))
def external():
 q=load(OUT/'queue_status.json');primary=sum(x.get('state') in ('train','test') for x in q['jobs'].values());u=load(OUT/'UNet2D/queue_status.json');c=load(OUT/'CMUNeXt/queue_status.json');active=('preflight','train','test');return primary+int(u.get('status') in active)+int(c.get('status') in active),c.get('status') in (*active,'complete','failed')
def start(name,phase,job):
 directory=O/'extra_logs'/name;directory.mkdir(parents=True,exist_ok=True);log=(directory/(phase+'.log')).open('x');env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=GPU,EXPERIMENT_PHYSICAL_GPU='1',EXPERIMENT_GPU_UUID=GPU,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1');cmd=[PY,'-B','-u',*job['phases'][phase]];p=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT);log.close();live[name]=(p,phase);states[name]=phase;save()
with (O/'extras_queue.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  while True:
   jobs=load(CFG)['jobs']
   for job in jobs:states.setdefault(job['name'],'waiting')
   lookup={x['name']:x for x in jobs}
   for name,(p,phase) in list(live.items()):
    rc=p.poll()
    if rc is None:continue
    del live[name];job=lookup[name]
    if rc:states[name]=phase+'_failed_exit_'+str(rc)
    elif phase=='preflight':states[name]='ready_to_train'
    elif phase=='train':states[name]='ready_to_test'
    else:
     done=Path(job['completion']);states[name]='complete' if done.is_file() and load(done).get('complete') is True else 'test_failed_missing_completion'
    save()
   used,previous_started=external()
   for job in jobs:
    name=job['name'];state=states[name]
    if used+len(live)>=2:break
    if not previous_started:break
    if state not in ('waiting','ready_to_train','ready_to_test'):continue
    if free()<int(job.get('min_free_mib',8192)):continue
    phase={'waiting':'preflight','ready_to_train':'train','ready_to_test':'test'}[state]
    bad=[path for path,digest in job.get('source_sha256',{}).items() if not Path(path).is_file() or hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest]
    if bad:
     states[name]='preflight_failed_source_drift'; (O/'extra_logs'/f'{name}_source_drift.json').parent.mkdir(parents=True,exist_ok=True); (O/'extra_logs'/f'{name}_source_drift.json').write_text(json.dumps(bad,indent=2)); save(); continue
    start(name,phase,job);time.sleep(8)
   save();time.sleep(15)
 except BaseException:
  (O/'extras_error.log').write_text(traceback.format_exc());raise

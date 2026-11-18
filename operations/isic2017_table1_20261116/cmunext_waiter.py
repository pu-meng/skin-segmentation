import fcntl,json,os,subprocess,time,traceback
from pathlib import Path
R=Path('/home/PuMengYu/experiments/skin_segmentation');O=R/'log/ISIC2017_表1_去重_20261116';runner=R/'operations/isic2017_table1_20261116/runner_cmunext.py';P='/home/PuMengYu/anaconda3/envs/medseg/bin/python';gpu='GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b';folder=O/'CMUNeXt';status=folder/'queue_status.json'
def update(state,**kw):
 t=status.with_suffix('.tmp');t.write_text(json.dumps({'status':state,'controller_pid':os.getpid(),'updated_at':time.time(),**kw},ensure_ascii=False,indent=2));t.replace(status)
def free():
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader'],text=True).splitlines();return int(next(x.split(',')[1].strip().split()[0] for x in rows if x.startswith(gpu)))
def alive(pid):
 try:os.kill(pid,0);return True
 except ProcessLookupError:return False
with (folder/'queue.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  while True:
   q=json.loads((O/'queue_status.json').read_text());p=q['controller_pid'];count=sum(1 for j in q['jobs'].values() if j.get('pid') and alive(j['pid']))
   if not alive(p):update('waiting_controller_missing');time.sleep(30);continue
   u=json.loads((O/'UNet2D/queue_status.json').read_text()); us=u.get('status',''); unet_active=us in ('preflight','train','test'); unet_started=unet_active or us in ('complete','failed'); primary_active=sum(j.get('state') in ('train','test') for j in q['jobs'].values())
   if unet_started and primary_active + int(unet_active) < 2 and free()>=6144:break
   update('waiting_for_slot',active_primary_jobs=count,free_mib=free());time.sleep(20)
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,EXPERIMENT_PHYSICAL_GPU='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
  for phase in ('preflight','train','test'):
   with (folder/(phase+'.log')).open('x') as log:
    proc=subprocess.Popen([P,'-B','-u',str(runner),phase,'CMUNeXt'],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT);update(phase,child_pid=proc.pid);rc=proc.wait()
   if rc:raise RuntimeError(f'{phase} exited {rc}')
  update('complete')
 except BaseException as e:
  update('failed',error=repr(e),traceback=traceback.format_exc());raise

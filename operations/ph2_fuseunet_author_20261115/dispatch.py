import fcntl,json,os,subprocess,time,traceback
from pathlib import Path
R=Path('/home/PuMengYu/experiments/skin_segmentation');O=R/'operations/ph2_fuseunet_author_20261115';PRE=R/'operations/ph2_s3hnet_author_20261115';OUT=Path('/media/8T/PuMengYu/experiments/skin_segmentation/PH2_FUSEUNET_AUTHOR_seed42_20261115');P='/home/PuMengYu/anaconda3/envs/medseg/bin/python';UUID='GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde'
def state(s,**kw):
 p=O/'queue_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(status=s,pid=os.getpid(),updated_at=time.time(),**kw),ensure_ascii=False,indent=2)+'\n');t.replace(p)
def free():
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader'],text=True).splitlines();return int(next(x.split(',')[1].strip().split()[0] for x in rows if x.startswith(UUID)))
def run(file,phase):
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=UUID,EXPERIMENT_GPU_UUID=UUID,EXPERIMENT_PHYSICAL_GPU='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_THREADING_LAYER='GNU');log=O/(phase+'.log')
 with log.open('x') as f:
  p=subprocess.Popen([P,'-B','-u',str(O/file),*([phase] if file=='runner.py' else [])],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT);state(phase,child_pid=p.pid,gpu=0);rc=p.wait()
 if rc:raise RuntimeError(f'{phase} failed ({rc}), see {log}')
try:
 with (O/'queue.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  while True:
   status=json.loads((PRE/'queue_status.json').read_text())
   if status['status']=='complete':break
   if status['status']=='failed':raise RuntimeError('S3H-Net predecessor failed')
   state('waiting_S3HNet',predecessor=status['status']);time.sleep(20)
  while free()<4096:state('waiting_gpu',free_mib=free(),required_mib=4096);time.sleep(20)
  with (R/'operations/encoder_swap_3slot_takeover_20261108/admission.lock').open('a') as admission:
   fcntl.flock(admission,fcntl.LOCK_EX)
   if free()<4096:raise RuntimeError('GPU memory changed before admission; retry required')
   run('runner.py','preflight')
  run('runner.py','train');run('runner.py','test');state('complete')
except BaseException as e:
 state('failed',error=repr(e),traceback=traceback.format_exc());raise

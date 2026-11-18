#!/usr/bin/env python3
"""Run one fixed-checkpoint retest without stopping existing training."""
import fcntl,json,os,subprocess,time,traceback
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation');OPS=ROOT/'operations/focaltransnet_isic_audit_20261114'
UUID='GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b';PY='/home/PuMengYu/anaconda3/envs/medseg/bin/python'
def state(status,**kw):
 p=OPS/'status.json';t=p.with_suffix(f'.{os.getpid()}.tmp');t.write_text(json.dumps(dict(status=status,pid=os.getpid(),updated_at=time.time(),**kw),ensure_ascii=False,indent=2)+'\n');t.replace(p)
def main():
 with (OPS/'test.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   while True:
    out=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader'],text=True)
    free=next((int(x.split(',')[1].strip().split()[0]) for x in out.splitlines() if x.split(',')[0].strip()==UUID),None)
    if free is not None and free>=2560:break
    state('waiting_gpu',free_mib=free,required_mib=2560);time.sleep(30)
   env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=UUID,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_THREADING_LAYER='GNU')
   with (OPS/'test.log').open('x') as log:
    child=subprocess.Popen([PY,'-B','-u',str(OPS/'test.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    state('testing',child_pid=child.pid,gpu=1);rc=child.wait()
   if rc:raise RuntimeError(f'test exited {rc}; see {OPS / "test.log"}')
   state('complete')
  except BaseException as exc:
   state('failed',error=repr(exc),traceback=traceback.format_exc());raise
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Train and test SkinFormer local adaptation in separate processes."""
import fcntl,json,os,subprocess,time,traceback
from pathlib import Path
R=Path('/home/PuMengYu/experiments/skin_segmentation');O=R/'operations/ph2_skinformer_20261115';OUT=Path('/media/8T/PuMengYu/experiments/skin_segmentation/PH2_SKINFORMER_LOCAL_S42_FRESH300_20261115');PY='/home/PuMengYu/anaconda3/envs/medseg/bin/python';ID='PH2_SKINFORMER_LOCAL_S42_FRESH300_20261115'
def write(status,**kw):
 p=O/'queue_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(status=status,pid=os.getpid(),updated_at=time.time(),**kw),ensure_ascii=False,indent=2)+'\n');t.replace(p)
def update(status,note,pid=None):
 cmd=[PY,str(R/'operations/expctl.py'),'update','--id',ID,'--status',status,'--gpu','0','--wait-reason','','--note',note]
 if pid is not None:cmd+=['--pid',str(pid)]
 subprocess.run(cmd,cwd=R,check=True,stdout=subprocess.DEVNULL)
def main():
 with (O/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   assert json.loads((O/'gpu_preflight.json').read_text())['passed']
   env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
   for mode in ('train','test'):
    with (O/(mode+'.log')).open('x') as log:
     p=subprocess.Popen([PY,'-B','-u',str(O/'run.py'),mode,'--out',str(OUT)],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
     write(mode,child_pid=p.pid,gpu=0);update('training' if mode=='train' else 'testing',f'SkinFormer local adaptation {mode}',p.pid)
     if p.wait()!=0:raise RuntimeError(f'{mode} failed; inspect {O/(mode+".log")}')
   assert json.loads((OUT/'test_complete.json').read_text())['complete'];assert all(json.loads((OUT/'final_test'/m/'artifact_audit.json').read_text())['complete'] for m in ('none','mirror4'));write('complete');update('complete','300轮及两种测试审计完成')
  except BaseException as e:
   write('failed',error=repr(e),traceback=traceback.format_exc());update('failed',repr(e));raise
if __name__=='__main__':main()

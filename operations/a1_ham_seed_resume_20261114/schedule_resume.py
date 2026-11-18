#!/usr/bin/env python3
"""Serially resume the two audited HAM seed runs without overtaking live queues."""
import fcntl
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
OPS=ROOT/'operations/a1_ham_seed_resume_20261114'
PY='/home/PuMengYu/anaconda3/envs/medseg/bin/python'
UUIDS=['GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde','GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b']
IDS={'a143':'A1_SWIFTFORMER_HAM_SEED43_20261113',
     'esd40':'A1_ESD_SINGLE_HAM_SEED40_RESUME_20261114'}
def read(path):
 try:return json.loads(Path(path).read_text())
 except FileNotFoundError:return {}
def status(state,case,**extra):
 path=OPS/'queue_status.json';tmp=path.with_suffix(f'.{os.getpid()}.tmp')
 tmp.write_text(json.dumps(dict(status=state,case=case,pid=os.getpid(),updated_at=time.time(),**extra),ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
def update(case,state,note,pid=None,gpu=None):
 cmd=[PY,str(ROOT/'operations/expctl.py'),'update','--id',IDS[case],'--status',state,'--note',note,'--wait-reason','']
 if pid is not None:cmd+=['--pid',str(pid)]
 if gpu is not None:cmd+=['--gpu',str(gpu)]
 subprocess.run(cmd,cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
def priority_released(gpu):
 if gpu==0:
  state=read(ROOT/'operations/focaltransnet_ham_20261114/queue_status.json')
  return state.get('status') in ('train','test','complete','failed')
 state=read(ROOT/'operations/a1_swift_skip_position_20261114/queue_status.json')
 return state.get('status') in ('complete','failed') or (state.get('position')==2 and state.get('status') in ('train','test'))
def select_gpu(case):
 while True:
  raw=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader'],text=True)
  free={row.split(',')[0].strip():int(row.split(',')[1].strip().split()[0]) for row in raw.splitlines()}
  ready=[g for g in (0,1) if priority_released(g) and free.get(UUIDS[g],0)>=4096]
  if ready:return max(ready,key=lambda g:free[UUIDS[g]])
  status('waiting_gpu',case,free_mib={str(g):free.get(UUIDS[g]) for g in (0,1)},required_mib=4096,
         priority='GPU0先让Focal HAM进入训练；GPU1先让skip2进入训练')
  time.sleep(30)
def phase(case,kind,gpu):
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=UUIDS[gpu],EXPERIMENT_PHYSICAL_GPU=str(gpu),
      EXPERIMENT_GPU_UUID=UUIDS[gpu],OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',
      MKL_THREADING_LAYER='GNU',PYTHONUNBUFFERED='1')
 with (OPS/f'{case}_{kind}.log').open('x') as log:
  child=subprocess.Popen([PY,'-B','-u',str(OPS/'worker.py'),kind,case,str(gpu)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
  update(case,'training' if kind=='resume' else 'testing',f'HAM seed恢复：{kind}，GPU{gpu}',child.pid,gpu)
  status('training' if kind=='resume' else 'testing',case,child_pid=child.pid,gpu=gpu)
  rc=child.wait()
 if rc:raise RuntimeError(f'{case} {kind} failed ({rc}); see {OPS / (case+"_"+kind+".log")}')
def main():
 with (OPS/'queue.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  case='a143'
  try:
   for case in ('a143','esd40'):
    update(case,'queued','断点核查通过，等待GPU；不抢占已有队列',os.getpid())
   for case in ('a143','esd40'):
    gpu=select_gpu(case);phase(case,'resume',gpu)
    gpu=select_gpu(case);phase(case,'test',gpu)
    update(case,'complete','HAM 300轮及none/mirror4测试与审计完成',gpu=gpu)
   status('complete',case)
  except BaseException as exc:
   status('failed',case,error=repr(exc),traceback=traceback.format_exc())
   update(case,'failed',f'恢复队列停止：{exc}')
   other='esd40' if case=='a143' else None
   if other:update(other,'stalled','前序恢复失败，未启动；检查恢复队列日志')
   raise
if __name__=='__main__':main()

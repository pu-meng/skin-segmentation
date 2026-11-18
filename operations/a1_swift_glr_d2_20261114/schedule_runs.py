#!/usr/bin/env python3
"""C0 then C1, after the previously scheduled skip2 run and tests."""
import fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
OPS=ROOT/'operations/a1_swift_glr_d2_20261114'
PY='/home/PuMengYu/anaconda3/envs/medseg/bin/python'
UUID='GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b'
PREVIOUS=ROOT/'operations/a1_swift_skip_position_20261114/queue_status.json'
PREVIOUS_WORK=Path('/media/8T/PuMengYu/experiments/skin_segmentation/A1_SWIFTFORMER_SKIP2_ISIC_SEED42_FRESH300_20261114/ISIC2018/A1-skip2')
sys.path.insert(0,str(ROOT))
from operations.a1_swift_glr_d2_20261114 import entry
def read(p):
 try:return json.loads(Path(p).read_text())
 except FileNotFoundError:return {}
def state(status,variant,**kw):
 p=OPS/'queue_status.json';tmp=p.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(dict(status=status,variant=variant,pid=os.getpid(),updated_at=time.time(),**kw),ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def update(variant,status,note,pid=None):
 cmd=[PY,str(ROOT/'operations/expctl.py'),'update','--id',f'A1_SWIFTFORMER_GLR_D2_{variant}_ISIC_SEED42_FRESH300_20261114','--status',status,'--gpu','1','--wait-reason','','--note',note]
 if pid is not None:cmd+=['--pid',str(pid)]
 subprocess.run(cmd,cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
def previous_live(pid):
 try:
  cmd=Path(f'/proc/{int(pid)}/cmdline').read_bytes().replace(b'\0',b' ').decode()
  return 'operations/a1_swift_skip_position_20261114/schedule_runs.py' in cmd
 except (ValueError,TypeError,FileNotFoundError,ProcessLookupError):return False
def wait_previous():
 while True:
  old=read(PREVIOUS)
  done=(old.get('status')=='complete' and read(PREVIOUS_WORK/'completion.json').get('complete') is True
        and all(read(PREVIOUS_WORK/'final_test'/mode/'artifact_audit.json').get('complete') is True for mode in ('none','mirror4')))
  if done:return
  if old.get('status')=='failed' or not previous_live(old.get('pid')):
   raise RuntimeError(f'skip2前序未完成且控制器不在运行: {old}')
  state('waiting_skip2','C0',predecessor_status=old.get('status'),predecessor_pid=old.get('pid'))
  time.sleep(30)
def wait_gpu(variant):
 while True:
  rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader'],text=True)
  free=next((int(x.split(',')[1].strip().split()[0]) for x in rows.splitlines() if x.split(',')[0].strip()==UUID),None)
  if free is not None and free>=4096:return
  state('waiting_gpu',variant,free_mib=free,required_mib=4096);time.sleep(30)
def main():
 with (OPS/'queue.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);variant='C0'
  try:
   for variant in ('C0','C1'):update(variant,'queued','CPU验收通过；等待skip2完成训练及双模式审计，随后真实batch8预检',os.getpid())
   variant='C0';wait_previous()
   for variant in ('C0','C1'):
    work=entry.run_dir(variant)/'ISIC2018'/entry.arm(variant)
    for kind in ('preflight','train','test'):
     wait_gpu(variant)
     env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=UUID,EXPERIMENT_PHYSICAL_GPU='1',EXPERIMENT_GPU_UUID=UUID,
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_THREADING_LAYER='GNU',PYTHONUNBUFFERED='1')
     with (work/(kind+'.log')).open('x') as log:
      child=subprocess.Popen([PY,'-B','-u',str(OPS/'entry.py'),'phase',variant,kind,'1'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
      update(variant,{'preflight':'preflight','train':'training','test':'testing'}[kind],f'{kind} on GPU1',child.pid)
      state(kind,variant,child_pid=child.pid,gpu=1);code=child.wait()
     if code:raise RuntimeError(f'{variant} {kind} failed {code}; see {work / (kind+".log")}')
    if not read(work/'completion.json').get('complete'):raise RuntimeError('completion audit absent')
    update(variant,'complete','300轮、none/mirror4均完成且审计通过')
   state('complete','C1')
  except BaseException as exc:
   state('failed',variant,error=repr(exc),traceback=traceback.format_exc());update(variant,'failed',repr(exc))
   if variant=='C0':update('C1','stalled','C0失败或前序中断，未启动；查看GLR队列日志')
   raise
if __name__=='__main__':main()

import fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation'); OPS=ROOT/'operations/a1_swift_glr_d2_c2c3_20261114'; PY='/home/PuMengYu/anaconda3/envs/medseg/bin/python'; UUID='GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde'
v=sys.argv[1]; assert v in ('C2','C3')
name=f'A1_SWIFTFORMER_GLR_D2_{v}_ISIC_SEED42_FRESH300_20261114'; work=Path('/media/8T/PuMengYu/experiments/skin_segmentation')/name/'ISIC2018'/f'A1-GLR-D2-{v}'
def status(s,**kw):
 p=OPS/f'{v}_status.json'; t=p.with_suffix('.tmp'); t.write_text(json.dumps(dict(status=s,variant=v,controller_pid=os.getpid(),time=time.time(),**kw),ensure_ascii=False,indent=2)+'\n');t.replace(p)
def update(s,note,pid=None):
 cmd=[PY,str(ROOT/'operations/expctl.py'),'update','--id',name,'--status',s,'--gpu','0','--wait-reason','','--note',note]
 if pid:cmd+=['--pid',str(pid)]
 subprocess.run(cmd,cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
def free():
 s=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader'],text=True)
 return next(int(r.split(',')[1].strip().split()[0]) for r in s.splitlines() if r.startswith(UUID))
def gpu_pids():
 s=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 return [int(r.split(',')[1]) for r in s.splitlines() if r.startswith(UUID)]
try:
 for kind in ('preflight','train','test'):
  admission=None
  if kind!='test':
   while True:
    admission=(ROOT/'operations/encoder_swap_3slot_takeover_20261108/admission.lock').open('a');fcntl.flock(admission,fcntl.LOCK_EX)
    available=free()
    if available>=4096:break
    fcntl.flock(admission,fcntl.LOCK_UN);admission.close();admission=None
    status('waiting_gpu',phase=kind,free_mib=available);update('waiting_gpu',f'{kind} 等待 GPU0 空闲至少4096 MiB',os.getpid());time.sleep(20)
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=UUID,EXPERIMENT_PHYSICAL_GPU='0',EXPERIMENT_GPU_UUID=UUID,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_THREADING_LAYER='GNU',PYTHONUNBUFFERED='1')
  with (work/(kind+'.log')).open('x') as log:
   child=subprocess.Popen([PY,'-B','-u',str(OPS/'entry.py'),'phase',v,kind,'0'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
   status(kind,child_pid=child.pid,free_before_mib=free());update({'preflight':'preflight','train':'training','test':'testing'}[kind],f'GPU0 {kind}',child.pid)
   if admission:
    deadline=time.time()+180
    while child.poll() is None and child.pid not in gpu_pids():
     if time.time()>deadline:child.terminate();raise TimeoutError('GPU admission timeout')
     time.sleep(1)
    fcntl.flock(admission,fcntl.LOCK_UN);admission.close();admission=None
   code=child.wait()
  if code:raise RuntimeError(f'{kind} failed {code}; see {work / (kind+".log")}')
 assert json.loads((work/'completion.json').read_text())['complete']
 update('complete','训练、无TTA/TTA4和审计完成');status('complete')
except BaseException as exc:
 if 'admission' in globals() and admission: fcntl.flock(admission,fcntl.LOCK_UN);admission.close()
 status('failed',error=repr(exc),traceback=traceback.format_exc());update('failed',repr(exc));raise

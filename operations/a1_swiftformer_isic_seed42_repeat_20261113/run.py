"""Resource-gated launcher for the seed42 A1-only ISIC repeat."""
import fcntl, json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
OPS=Path(__file__).resolve().parent
ENTRY=OPS/'entry.py'
RUN=Path('/media/8T/PuMengYu/experiments/skin_segmentation/A1_SWIFTFORMER_ISIC_SEED42_REPEAT_FRESH300_20261113')
PY='/home/PuMengYu/anaconda3/envs/medseg/bin/python'
UUIDS=['GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde','GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b']
GUARD=ROOT/'operations/encoder_swap_3slot_takeover_20261108/admission.lock'
def write(path,obj):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)); tmp.replace(path)
def gpu_state(gpu):
 out=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 pids=[int(x.split(',')[1]) for x in out.splitlines() if x.startswith(UUIDS[gpu])]
 stats=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader'],text=True)
 free=int(next(x.split(',')[1].strip().split()[0] for x in stats.splitlines() if x.startswith(UUIDS[gpu])))
 return pids,free
def worker(gpu):
 import importlib.util
 sys.path.insert(0,str(ROOT))
 spec=importlib.util.spec_from_file_location('seed42_repeat_entry',ENTRY)
 entry=importlib.util.module_from_spec(spec); spec.loader.exec_module(entry)
 work=RUN/'A1'
 with (work/'repeat_controller.lock').open('a') as local:
  fcntl.flock(local,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for mode in ('preflight','train','test'):
   entry.verify()
   while True:
    gate=GUARD.open('a'); fcntl.flock(gate,fcntl.LOCK_EX)
    pids,free=gpu_state(gpu)
    if len(pids)<4 and free>=8192: break
    write(RUN/'swiftformer_s_waiting.json',dict(status='waiting_for_gpu_admission',gpu=gpu,compute_pids=pids,free_mib=free,required_free_mib=8192,updated_unix=time.time()))
    fcntl.flock(gate,fcntl.LOCK_UN); gate.close(); time.sleep(30)
   env=os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=UUIDS[gpu],EXPERIMENT_PHYSICAL_GPU=str(gpu),EXPERIMENT_GPU_UUID=UUIDS[gpu],OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
   log=(work/(mode+'.log')).open('x')
   child=subprocess.Popen([PY,'-B','-u',str(ENTRY),mode,'A1',str(gpu)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   write(work/'controller_status.json',dict(status=mode,controller_pid=os.getpid(),child_pid=child.pid,physical_gpu=gpu,gpu_uuid=UUIDS[gpu],updated_unix=time.time()))
   write(work/(mode+'_status.json'),dict(status=mode,pid=child.pid,gpu=gpu,started_unix=time.time()))
   deadline=time.time()+300
   while child.poll() is None:
    active,_=gpu_state(gpu)
    if child.pid in active: break
    if time.time()>deadline:
     child.terminate(); raise TimeoutError('phase did not become GPU resident within 300 seconds')
    time.sleep(1)
   fcntl.flock(gate,fcntl.LOCK_UN); gate.close()
   rc=child.wait(); log.close(); write(work/(mode+'_exitcode.json'),dict(exitcode=rc,pid=child.pid,finished_unix=time.time()))
   if rc: raise RuntimeError(f'{mode} failed; see {work/(mode+".log")}')
   try: (RUN/'swiftformer_s_waiting.json').unlink()
   except FileNotFoundError: pass
  assert entry.read(RUN/'A1/completion.json')['complete']
  write(work/'controller_status.json',dict(status='complete',controller_pid=os.getpid(),physical_gpu=gpu,updated_unix=time.time()))
if __name__=='__main__': worker(int(sys.argv[1]))

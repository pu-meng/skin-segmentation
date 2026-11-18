import fcntl,json,os,select,subprocess,sys,time
import ctypes,hashlib
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
RUN=Path('/media/8T/PuMengYu/experiments/skin_segmentation/A1Swift_PairAblations_ISIC_FRESH300_20261112')
PY='/home/PuMengYu/anaconda3/envs/medseg/bin/python'
ARM='no_skip1_no_e2';GPU=1;PID=1450555;UUID='GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b'
def write(p,x):
 p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');t.replace(p)
def ready():
 p=json.loads((RUN/'protocol.json').read_text())
 for f,h in p['protected_sha256'].items():
  import hashlib
  z=hashlib.sha256(Path(f).read_bytes()).hexdigest();assert z==h,'source drift '+f
 return p

def main():
 st=RUN/ARM/'test_recovery_status.json';deadline=time.time()+8*3600
 write(st,dict(status='waiting_for_existing_train_process',pid=PID,arm=ARM,gpu=GPU,deadline_unix=deadline))
 if hasattr(os,'pidfd_open'):fd=os.pidfd_open(PID)
 else:
  fd=ctypes.CDLL(None,use_errno=True).syscall(434,PID,0)
  if fd<0:raise OSError(ctypes.get_errno(),'pidfd_open failed')
 poll=select.poll();poll.register(fd,select.POLLIN)
 poll.poll(max(1,int((deadline-time.time())*1000)));os.close(fd)
 sys.path[:0]=[str(ROOT),str(ROOT/'source/bottleneck_mamba')]
 p=ready()
 from final_test_pipeline import training_ready
 assert training_ready(RUN/ARM/'run'),'training process exited without complete audited run'
 train_prov=json.loads((RUN/ARM/'run/provenance.json').read_text())
 write(st,dict(status='training_complete_waiting_test_gpu',pid=PID,arm=ARM,gpu=GPU,training_protocol_sha256=train_prov['protocol_sha256'],test_protocol_sha256=hashlib.sha256((RUN/'protocol.json').read_bytes()).hexdigest(),updated_unix=time.time()))
 gate=p['scheduling']['free_vram_gate_mib'];guard=ROOT/'operations/encoder_swap_3slot_takeover_20261108/admission.lock'
 while time.time()<deadline:
  lock=guard.open('a');fcntl.flock(lock,fcntl.LOCK_EX)
  rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
  pids=[int(x.split(',')[1]) for x in rows.splitlines() if x.startswith(UUID)]
  rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader'],text=True)
  free=int(next(x.split(',')[1].strip().split()[0] for x in rows.splitlines() if x.startswith(UUID)))
  if len(pids)<4 and free>=gate:break
  fcntl.flock(lock,fcntl.LOCK_UN);lock.close();write(st,dict(status='waiting_test_gpu',arm=ARM,gpu=GPU,free_mib=free,threshold_mib=gate,compute_pids=pids,updated_unix=time.time()));time.sleep(30)
 else:raise TimeoutError('test GPU admission deadline')
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=UUID,EXPERIMENT_PHYSICAL_GPU=str(GPU),EXPERIMENT_GPU_UUID=UUID,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 log=(RUN/ARM/'test.log').open('x');write(st,dict(status='test_started',arm=ARM,gpu=GPU,pid=PID,free_mib_at_admission=free,training_protocol_sha256=train_prov['protocol_sha256'],test_protocol_sha256=hashlib.sha256((RUN/'protocol.json').read_bytes()).hexdigest(),updated_unix=time.time()))
 child=subprocess.Popen([PY,'-B','-u',str(ROOT/'operations/a1swift_pair_ablation_20261112/entry.py'),'phase',ARM,'test',str(GPU)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 fcntl.flock(lock,fcntl.LOCK_UN);lock.close();rc=child.wait();log.close();write(RUN/ARM/'test_exit.json',dict(exitcode=rc,pid=child.pid,time=time.time()));assert rc==0,'test phase failed'
 write(st,dict(status='test_complete',arm=ARM,gpu=GPU,pid=child.pid,updated_unix=time.time()))
 print('RECOVERED_TEST_COMPLETE',ARM,flush=True)
if __name__=='__main__':main()

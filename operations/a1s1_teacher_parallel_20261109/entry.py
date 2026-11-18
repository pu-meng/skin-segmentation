"""Run already authorized KD arms on GPU0 without restarting the active student."""
import fcntl, json, os, subprocess, sys, time, traceback
from pathlib import Path

ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
RUN=Path('/media/8T/PuMengYu/experiments/skin_segmentation/A1S1_TEACHER_HAM_ISIC_20261109')
A1=Path('/media/8T/PuMengYu/experiments/skin_segmentation/A1_Teacher_RMT-T_KD-V1_HAM_FRESH300_20261109_r2/HAM10000')
MAIN=ROOT/'operations/a1s1_teacher_20261109/entry.py'
A1_MAIN=ROOT/'operations/a1_teacher_kd_v1_r2_20261109/entry.py'
PY='/home/PuMengYu/anaconda3/envs/medseg/bin/python'
GPU='GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde'
JOBS=('SKIP_ISIC2018','S1_HAM10000','SKIP_HAM10000')

def read(path):
 try:return json.loads(Path(path).read_text())
 except (OSError,ValueError):return {}
def write(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
def command(pid):
 try:
  proc=Path('/proc')/str(int(pid));state=(proc/'stat').read_text().rsplit(')',1)[1].split()[0]
  return '' if state=='Z' else (proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
 except (OSError,ValueError,TypeError,IndexError):return ''
def resources():
 s=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True)
 return int(next(line.split(',')[1].strip() for line in s.splitlines() if line.startswith(GPU)))
def launch(script,args,log,status):
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=GPU,EXPERIMENT_PHYSICAL_GPU='0',EXPERIMENT_GPU_UUID=GPU,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 gate=ROOT/'operations/encoder_swap_3slot_takeover_20261108/admission.lock'
 while True:
  with gate.open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   if resources()<4500:
    write(status,{'status':'waiting_gpu_memory','free_mib':resources(),'controller_pid':os.getpid(),'gpu':0});continue_wait=True
   else:
    continue_wait=False
    with log.open('x') as stream:
     child=subprocess.Popen([PY,'-B','-u',str(script),*args],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
     write(status,{'status':args[-1] if script==MAIN else 'test','child_pid':child.pid,'controller_pid':os.getpid(),'gpu':0})
     deadline=time.time()+120
     while child.poll() is None and time.time()<deadline:
      apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split()
      if str(child.pid) in apps:break
      time.sleep(1)
     else:
      if child.poll() is None:raise RuntimeError('child did not attach to GPU in 120 seconds')
  if continue_wait:time.sleep(15);continue
  rc=child.wait();write(log.with_suffix('.exit.json'),{'exitcode':rc,'pid':child.pid});assert rc==0,(str(log),rc)
  return
def run_job(name):
 work=RUN/name;work.mkdir(exist_ok=True);status=work/'parallel_status.json'
 with (work/'parallel.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert not (work/'run').exists() and not (work/'preflight').exists()
  for kind in ('preflight','train','test'):launch(MAIN,['phase',name,kind],work/('parallel_'+kind+'.log'),status)
  assert read(work/'completion.json').get('complete') is True
  write(status,{'status':'complete','controller_pid':os.getpid(),'gpu':0})
def wait_active(pid):
 work=RUN/'S1_ISIC2018';status=work/'parallel_status.json';expected='a1s1_teacher_20261109/entry.py phase S1_ISIC2018 train'
 with (work/'parallel.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  write(status,{'status':'waiting_existing_train','child_pid':pid,'controller_pid':os.getpid(),'gpu':0})
  while expected in command(pid):time.sleep(15)
  audit=read(work/'run/artifact_audit.json');assert audit.get('training_complete') and audit.get('epochs')==300 and audit.get('validation_artifacts_complete')
  launch(MAIN,['phase','S1_ISIC2018','test'],work/'parallel_test.log',status)
  assert read(work/'completion.json').get('complete') is True
  write(status,{'status':'complete','controller_pid':os.getpid(),'gpu':0})
def test_a1():
 with (A1/'manual_test.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  audit=read(A1/'run/artifact_audit.json');assert audit.get('training_complete') and audit.get('validation_artifacts_complete') and audit.get('epochs')==300
  assert not (A1/'final_test').exists() and not (A1/'completion.json').exists()
  launch(A1_MAIN,['phase','test','0'],A1/'manual_test_20261109.log',A1/'manual_test_status.json')
  assert read(A1/'completion.json').get('complete') is True
  write(A1/'manual_test_status.json',{'status':'complete','controller_pid':os.getpid(),'gpu':0})
if __name__=='__main__':
 action=sys.argv[1]
 try:
  if action=='job' and sys.argv[2] in JOBS:run_job(sys.argv[2])
  elif action=='watch':wait_active(int(sys.argv[2]))
  elif action=='a1test':test_a1()
  else:raise ValueError('unknown action')
 except BaseException as e:
  dest=(A1/'manual_test_failure.json') if action=='a1test' else (RUN/(sys.argv[2] if action=='job' else 'S1_ISIC2018')/'parallel_failure.json')
  write(dest,{'error':repr(e),'traceback':traceback.format_exc(),'time':time.time()});raise

"""Complete the previously blocked single baseline; no old run overwrite."""
from pathlib import Path
import os,sys,subprocess,json,fcntl,hashlib,shutil
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'log/TinyViM_四层对应Decoder_拉普拉斯5_2_HAM_20261104'
if __name__=='__main__':
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if any((RUN/'run'/p).exists() for p in ['status.json','checkpoints','history.csv']):raise FileExistsError('Existing training')
  free=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
  if free<12288:raise RuntimeError('Less than 12 GiB free')
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',EXPERIMENT_PHYSICAL_GPU='0',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4')
  provenance={}
  for script in [Path(__file__).resolve(),Path(__file__).with_name('four_stage_laplacian_entry.py').resolve()]:
   dest=RUN/'launch_source_snapshot'/script.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(script,dest)
   provenance[str(script.relative_to(ROOT))]=hashlib.sha256(script.read_bytes()).hexdigest()
  state=dict(status='starting',controller_pid=os.getpid(),physical_gpu=0,source_sha256=provenance,training_source_unchanged=True)
  def save():
   tmp=RUN/'launch_status.tmp';tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n');tmp.replace(RUN/'launch_status.json')
  save()
  for phase in ['preflight','train','test']:
   with (RUN/f'gpu_{phase}_console.log').open('x') as stream:
    child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).with_name('four_stage_laplacian_entry.py')),'--phase',phase],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    state.update(status=phase,child_pid=child.pid);save();code=child.wait()
   if code:state.update(status=phase+'_failed',exitcode=code);save();sys.exit(code)
  state['status']='complete';save()

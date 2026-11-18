import os,sys,subprocess,json,fcntl,traceback,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
LOG=ROOT/'log/mambalite_fig2b_20261105'
PY='/home/PuMengYu/anaconda3/envs/medseg/bin/python'
LOG.mkdir(parents=True,exist_ok=True)
lock=(LOG/'controller.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state={'pid':os.getpid(),'status':'starting','stage':None,'seed':None,'started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
def save_state():
 p=LOG/'controller_resume_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(p)
def run(stage,seed=None):
 manifest=json.loads((HERE/'revision_manifest.json').read_text())
 import hashlib
 for name,digest in manifest['files'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,'Revision source changed: '+name
 state.update(stage=stage,seed=seed,status='running',updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'));save_state()
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='' if stage in ('cpu','strict') else '1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
 if stage in ('cpu','gpu'):cmd=[PY,'-u',str(HERE/'acceptance.py'),stage]
 else:cmd=[PY,'-u',str(HERE/'experiment_revision.py'),stage]+(['--seed',str(seed)] if seed is not None else [])
 dest=(LOG/f'seed{seed}' if seed is not None else LOG);dest.mkdir(parents=True,exist_ok=True)
 print('CONTROLLER RUN',stage,seed,' '.join(cmd),flush=True)
 with (dest/(stage+'_revision.log')).open('x',buffering=1) as log:
  p=subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
  state['child_pid']=p.pid;save_state()
  for line in p.stdout:
   log.write(line);print(line,end='',flush=True)
  rc=p.wait()
 (dest/(stage+'_revision.exitcode')).write_text(str(rc)+'\n')
 if rc:raise RuntimeError(f'{stage} seed={seed} failed exitcode={rc}; downstream blocked')
 print('CONTROLLER PASSED',stage,seed,flush=True)
try:
 (LOG/'controller_resume.pid').write_text(str(os.getpid())+'\n');save_state()
 assert (HERE/'boundary_stop.json').exists()
 assert not Path('/proc/2534485').exists()
 assert not Path('/proc/2534298').exists()
 for seed in range(42,47):
  run('train',seed);run('strict',seed);run('test',seed)
 run('aggregate')
 state.update(status='complete',child_pid=None);save_state()
except BaseException as e:
 state.update(status='failed_downstream_blocked',error=repr(e),traceback=traceback.format_exc(),child_pid=None);save_state()
 print('CONTROLLER BLOCKED',json.dumps(state),flush=True)
 raise

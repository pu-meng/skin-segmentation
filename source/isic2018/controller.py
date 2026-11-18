"""One sequential GPU queue: verify all12 preflights, then train and test each arm."""
from pathlib import Path
from datetime import datetime
import fcntl,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
from experiment import ARMS,LOG
from data import sha
PYTHON=Path('/home/PuMengYu/experiments/MambaLiteUNet/.venv/bin/python')
ENTRY=Path(__file__).parent/'experiment.py'

def write(p,v):
 temporary=p.with_suffix(p.suffix+'.tmp');temporary.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');temporary.replace(p)
def main():
 LOG.mkdir(exist_ok=True)
 with (LOG/'queue_owner.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  state={'status':'initializing','pid':os.getpid(),'physical_gpu':int(os.environ['EXPERIMENT_PHYSICAL_GPU']),'tmux_session':'isic2018_queue_gpu1','arms':ARMS,'preflight_passed':[],'completed_arms':[],'started_at':datetime.now().astimezone().isoformat()}
  def save(**changes):state.update(changes,updated_at=datetime.now().astimezone().isoformat());write(LOG/'queue_status.json',state)
  def run(mode,arm=None):
   cmd=[str(PYTHON),'-u',str(ENTRY),'--mode',mode]
   if arm:cmd+=['--arm',arm]
   save(status=mode,current_arm=arm)
   work=LOG/'runs'/arm if arm else LOG;work.mkdir(parents=True,exist_ok=True)
   path=work/(mode+'_console.log')
   with path.open('a') as out:
    proc=subprocess.Popen(cmd,stdout=out,stderr=subprocess.STDOUT,cwd=ROOT)
    save(child_pid=proc.pid,current_log=str(path));code=proc.wait()
   if code:raise RuntimeError(f'{mode} {arm} failed with exit {code}; see {path}')
  try:
   run('prepare')
   for arm in ARMS:
    p=LOG/'runs'/arm/'preflight/preflight_status.json'
    if not p.exists():run('preflight',arm)
    report=json.loads(p.read_text());assert report['status']=='passed' and report['protocol_sha256']==sha(LOG/'protocol.json')
    state['preflight_passed'].append(arm);save()
   for arm in ARMS:
    # Launch every new GPU stage only with at least8GiB free; never signal peer jobs.
    while True:
     free=int(subprocess.check_output(['nvidia-smi','-i',str(state['physical_gpu']),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
     if free>=8000:break
     save(status='waiting_for_gpu_memory',current_arm=arm,free_mib=free);time.sleep(30)
    work=LOG/'runs'/arm
    if (work/'run/status.json').exists():
     from final_test_pipeline import training_ready
     if not training_ready(work/'run'):raise RuntimeError(f'Partial run {arm}: explicit audited recovery required; no overwrite')
    else:run('train',arm)
    if not (work/'test_artifact_audit.json').exists():run('test',arm)
    assert json.loads((work/'test_artifact_audit.json').read_text())['complete']
    state['completed_arms'].append(arm);save()
   save(status='all_complete',current_arm=None,child_pid=None)
  except BaseException as e:
   save(status='failed',error=repr(e));raise
if __name__=='__main__':main()

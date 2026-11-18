"""Scheduling-only extension: adopt the current job and maintain three GPU slots.
The original training protocol, source hashes, preflights and model code are immutable.
"""
from pathlib import Path
from datetime import datetime
import csv,fcntl,json,os,signal,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
from experiment import ARMS,LOG,protocol
from data import sha
PYTHON='/home/PuMengYu/experiments/MambaLiteUNet/.venv/bin/python'
ENTRY=Path(__file__).parent/'experiment.py'

def write(p,v):
 temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');temp.replace(p)
def live(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0] not in ['Z','X']
 except FileNotFoundError:return False
def matching(pid,script):
 try:return script.encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
 except FileNotFoundError:return False

def main():
 protocol()
 adoption=json.loads((LOG/'parallel_adoption.json').read_text());gpu=int(adoption['physical_gpu'])
 with (LOG/'parallel_controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  old_controller=adoption['old_controller_pid'];adopted_pid=adoption['training_pid'];adopted_arm=adoption['arm']
  assert live(adopted_pid) and matching(adopted_pid,b'/source/isic2018/experiment.py'.decode())
  active={adopted_arm:{'mode':'train','pid':adopted_pid,'process':None,'stream':None,'adopted':True}}
  completed=[];failed={};pending=[a for a in ARMS if a!=adopted_arm]
  state={'status':'running_parallel','pid':os.getpid(),'parallel_jobs':3,'physical_gpu':gpu,'tmux_session':'isic2018_queue_gpu1','arms':ARMS,'preflight_passed':ARMS,'started_at':datetime.now().astimezone().isoformat(),'adopted_training_pid':adopted_pid,'old_controller_suspended_pid':old_controller}
  def save():
   state.update(active_jobs={a:{k:v for k,v in j.items() if k not in ['process','stream']} for a,j in active.items()},pending_arms=pending,completed_arms=completed,failed_arms=failed,updated_at=datetime.now().astimezone().isoformat())
   write(LOG/'queue_status.json',state)
  def spawn(arm,mode):
   work=LOG/'runs'/arm
   if mode=='train':assert not (work/'run/status.json').exists(),f'Refuse overwrite {arm}'
   if mode=='test':assert not (work/'test').exists(),f'Refuse overwrite test {arm}'
   stream=(work/(mode+'_console.log')).open('a')
   p=subprocess.Popen([PYTHON,'-u',str(ENTRY),'--mode',mode,'--arm',arm],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
   active[arm]={'mode':mode,'pid':p.pid,'process':p,'stream':stream,'adopted':False}
   print('START',mode,arm,'PID',p.pid,flush=True);save()
  save()
  while active or pending:
   for arm,job in list(active.items()):
    process=job['process']
    if process is None:
     if live(job['pid']):continue
     code=0
    else:
     code=process.poll()
     if code is None:continue
    if job['stream'] is not None:job['stream'].close()
    del active[arm]
    if job['adopted']:
     # The adopted training child has exited. Its suspended original scheduler
     # can now be retired without touching a live training or any peer process.
     if live(old_controller):
      assert matching(old_controller,'source/isic2018/controller.py')
      procstate=Path(f'/proc/{old_controller}/stat').read_text().split(') ',1)[1].split()[0]
      assert procstate in ['T','t'],procstate
      os.kill(old_controller,signal.SIGKILL)
     state['old_controller_retired']=True
    work=LOG/'runs'/arm
    try:
     if code:raise RuntimeError(f'{job["mode"]} exited{code}')
     if job['mode']=='train':
      from final_test_pipeline import training_ready
      if not training_ready(work/'run'):raise RuntimeError('Training/validation incomplete')
      spawn(arm,'test')
     else:
      if json.loads((work/'test_artifact_audit.json').read_text()).get('complete') is not True:raise RuntimeError('Test artifacts incomplete')
      completed.append(arm);print('COMPLETE',arm,flush=True)
    except Exception as e:
     failed[arm]={'mode':job['mode'],'error':repr(e),'pid':job['pid']};print('FAILED',arm,repr(e),flush=True)
    save()
   while len(active)<3 and pending:
    free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if free<8000:
     state['waiting_for_gpu_memory_mib']=free;save();break
    arm=pending.pop(0);report=json.loads((LOG/'runs'/arm/'preflight/preflight_status.json').read_text())
    assert report['status']=='passed' and report['protocol_sha256']==sha(LOG/'protocol.json')
    spawn(arm,'train')
    # Serialize process initialization so memory headroom accounts for the last launch.
    deadline=time.monotonic()+30
    while time.monotonic()<deadline:
     if active[arm]['process'].poll() is not None or (LOG/'runs'/arm/'run/status.json').exists():break
     time.sleep(.5)
   save()
   time.sleep(3)
  state.update(status='complete_with_failures' if failed else 'all_complete');save()
if __name__=='__main__':main()

import os,sys,subprocess,fcntl,json,time
from pathlib import Path
from datetime import datetime
os.environ.update(CUDA_VISIBLE_DEVICES='0',EXPERIMENT_PHYSICAL_GPU='0',PYTHONUNBUFFERED='1')
import bootstrap
from experiment import OUT,write
HERE=Path(__file__).parent
OUT.mkdir(exist_ok=True)
with (OUT/'controller.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state=dict(controller_pid=os.getpid(),gpu=0,shared_gpu_authorized=True)
 try:
  for name in ['cpu','gpu']:assert json.loads((HERE/(name+'_acceptance.json')).read_text())['passed']
  for phase in ['preflight','train','test']:
   log=OUT/(phase+'_console.log');assert not log.exists(),'Refuse phase log overwrite'
   with log.open('x') as f:
    child=subprocess.Popen([sys.executable,'-u',str(HERE/'experiment.py'),'--mode',phase],cwd=bootstrap.ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    state.update(status=phase,child_pid=child.pid,log=str(log),updated_at=datetime.now().astimezone().isoformat());write(OUT/'controller_status.json',state)
    # foreground output appears in tmux and persistent phase log
    for line in child.stdout:sys.stdout.write(line);sys.stdout.flush();f.write(line);f.flush()
    rc=child.wait();(OUT/(phase+'.exitcode')).write_text(str(rc)+'\n');assert rc==0,f'{phase} failed code {rc}; no retry'
  assert json.loads((OUT/'results.json').read_text())['complete']
  state.update(status='complete',updated_at=datetime.now().astimezone().isoformat());write(OUT/'controller_status.json',state);(OUT/'controller.exitcode').write_text('0\n')
 except BaseException as e:
  state.update(status='failed',error=repr(e),updated_at=datetime.now().astimezone().isoformat());write(OUT/'controller_status.json',state);(OUT/'controller.exitcode').write_text('1\n');raise

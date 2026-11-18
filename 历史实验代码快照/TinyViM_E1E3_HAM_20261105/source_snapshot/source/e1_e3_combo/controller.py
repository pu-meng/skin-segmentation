"""One approved arm: real-batch preflight -> fresh 300 epochs -> fixed-best test."""
from pathlib import Path
from datetime import datetime
import argparse,fcntl,json,os,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['HAM','ISIC'],required=True);p.add_argument('--arm',choices=['E1_E3','E0'],required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),EXPERIMENT_PHYSICAL_GPU=str(a.gpu),PYTHONUNBUFFERED='1')
from experiment import ROOTS,ARMS,write,ROOT
assert a.arm in ARMS[a.dataset]
work=ROOTS[a.dataset]/a.arm;work.mkdir(parents=True,exist_ok=True)
with (work/'controller.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state=dict(dataset=a.dataset,arm=a.arm,gpu=a.gpu,controller_pid=os.getpid())
 try:
  peak=json.loads(Path(__file__).with_name('gpu_E1_E3_acceptance.json').read_text())['peak_allocated']
  free=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>=peak/2**20+2048
  for phase in ['preflight','train','test']:
   path=work/(phase+'_console.log');assert not path.exists(),'Refusing phase-log overwrite'
   with path.open('x') as stream:
    child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).with_name('experiment.py')),'--dataset',a.dataset,'--arm',a.arm,'--mode',phase],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    state.update(status=phase,child_pid=child.pid,log=str(path),updated_at=datetime.now().astimezone().isoformat());write(work/'controller_status.json',state)
    code=child.wait()
   assert code==0,f'{phase} failed with {code}: {path}'
  audit=json.loads((work/'test_artifact_audit.json').read_text());assert audit['complete'] and audit['prediction_count']==(2020 if a.dataset=='HAM' else 518)
  state.update(status='complete',finished_at=datetime.now().astimezone().isoformat());write(work/'controller_status.json',state)
 except BaseException as error:
  state.update(status='failed',error=repr(error),updated_at=datetime.now().astimezone().isoformat());write(work/'controller_status.json',state);raise

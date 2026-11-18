"""Launch the user-approved experiment after verified preflight; retain all final artifacts."""
import os,sys,time,json,fcntl,subprocess
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
PYTHON='/home/PuMengYu/anaconda3/envs/medseg/bin/python'
GPU='1'

def read(path):return json.loads(path.read_text())
def save(row):
 row.update(updated_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),physical_gpu=GPU)
 p=ROOT/'pipeline_state.json';temp=p.with_suffix('.tmp');temp.write_text(json.dumps(row,indent=2));temp.replace(p)
 print(json.dumps(row),flush=True)
def free_gib():
 r=subprocess.run(['nvidia-smi','-i',GPU,'--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
 return float(r.stdout.strip())/1024

def main():
 os.environ.update(PYTHONPATH='/home/PuMengYu/nnUNet',CUDA_VISIBLE_DEVICES=GPU,
                   PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
 with (ROOT/'pipeline.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  from pumengyu.frequency_replay50.protocol import verify_protocol
  protocol=read(ROOT/'replay_fusion_protocol.json');verify_protocol(protocol)
  for kind in ['cpu','gpu']:
   if not read(ROOT/f'{kind}_preflight.json')['passed']:raise RuntimeError('Unverified '+kind+' preflight')
  check=read(ROOT/'gpu_preflight.json')
  need=max(10.,check['peak_reserved_gib']+1.)
  stages=[]
  commands=[('train',[PYTHON,'-u','-m','pumengyu.frequency_replay50.run','train','--gpu',GPU,'--min-free-gib',str(need)])]
  domains=['HCC','IRCADb','LiTS','validation','HCC_validation']
  commands += [(d,[PYTHON,'-u','-m','pumengyu.frequency_replay50.evaluate','--gpu',GPU,'--domain',d]) for d in domains]
  for stage,command in commands:
   required=need if stage=='train' else 8.
   while free_gib()<required:
    save(dict(status='waiting_for_memory',stage=stage,required_free_gib=required,stages=stages));time.sleep(20)
   with (ROOT/'logs'/f'{stage}.log').open('a') as stream:
    child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    save(dict(status='running',stage=stage,child_pid=child.pid,stages=stages,required_free_gib=required))
    code=child.wait()
   stages.append(dict(stage=stage,exit_code=code))
   if code:
    save(dict(status='failed',stage=stage,stages=stages));return code
  for domain in domains:
   if not read(ROOT/'evaluation'/domain/'artifact_audit.json')['complete']:raise RuntimeError('Incomplete '+domain)
  save(dict(status='complete',stages=stages));return 0
if __name__=='__main__':
 try:sys.exit(main())
 except Exception as error:
  save(dict(status='failed',error=repr(error)));raise

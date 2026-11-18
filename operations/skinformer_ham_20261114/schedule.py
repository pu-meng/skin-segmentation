#!/usr/bin/env python3
"""Start SkinFormer HAM after audited SkinFormer ISIC completion."""
import fcntl,hashlib,json,os,subprocess,time,traceback
from pathlib import Path
R=Path('/home/PuMengYu/experiments/skin_segmentation');O=R/'operations/skinformer_ham_20261114'; P=R/'operations/skinformer_adapt_20261114';PY='/home/PuMengYu/anaconda3/envs/medseg/bin/python';BASE=Path('/media/8T/PuMengYu/experiments/skin_segmentation');ISIC=BASE/'SKINFORMER_LOCAL_ISIC2018_S42_FRESH300_20261114';OUT=BASE/'SKINFORMER_LOCAL_HAM10000_S42_FRESH300_20261114';ID='SKINFORMER_LOCAL_HAM10000_S42_FRESH300_20261114'
def read(p):
 try:return json.loads(Path(p).read_text())
 except FileNotFoundError:return {}
def state(s,**kw):
 p=O/'queue_status.json';t=p.with_suffix(f'.{os.getpid()}.tmp');t.write_text(json.dumps({'status':s,'pid':os.getpid(),'updated_at':time.time(),**kw},ensure_ascii=False,indent=2)+'\n');t.replace(p)
def update(s,note,pid=None):
 cmd=[PY,str(R/'operations/expctl.py'),'update','--id',ID,'--status',s,'--gpu','0','--wait-reason','','--note',note]
 if pid:cmd+=['--pid',str(pid)]
 subprocess.run(cmd,cwd=R,check=True,stdout=subprocess.DEVNULL)
def live(pid):
 try:
  cmd=Path(f'/proc/{int(pid)}/cmdline').read_bytes().replace(b'\0',b' ').decode();stat=Path(f'/proc/{int(pid)}/stat').read_text().split(') ',1)[1][0]
  return stat!='Z' and 'skinformer_adapt_20261114/controller.py' in cmd
 except (FileNotFoundError,ProcessLookupError,TypeError,ValueError):return False
def verify():
 for path,expected in read(O/'prepared.json')['source_sha256'].items():
  h=hashlib.sha256()
  with open(path,'rb') as f:
   for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
  if h.hexdigest()!=expected:raise RuntimeError('source changed since prepare: '+path)
def main():
 with (O/'queue.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   while True:
    prev=read(P/'queue_status.json');status=prev.get('status');done=read(ISIC/'test_complete.json').get('complete') is True
    audits=all(read(ISIC/'final_test'/mode/'artifact_audit.json').get('complete') is True for mode in ('none','mirror4'))
    if status=='complete' and done and audits:break
    if status=='failed' or not live(prev.get('pid')):raise RuntimeError(f'ISIC predecessor failed or controller stopped before tests: {prev}')
    state('waiting_isic',predecessor_status=status,predecessor_pid=prev.get('pid'));time.sleep(30)
   while True:
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader'],text=True).splitlines();free=int(next(x.split(',')[1].strip().split()[0] for x in rows if x.split(',')[0].strip()=='0'))
    if free>=5120:break
    state('waiting_gpu',free_mib=free,required_mib=5120);time.sleep(30)
   verify();assert not OUT.exists(),'refusing existing output';env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
   phases=[('preflight',[PY,'-B','-u',str(O/'preflight.py')]),('train',[PY,'-B','-u',str(O/'run.py'),'train','--out',str(OUT)]),('test',[PY,'-B','-u',str(O/'run.py'),'test','--out',str(OUT)])]
   for mode,cmd in phases:
    verify()
    if mode=='train' and not read(O/'gpu_preflight.json').get('passed'):raise RuntimeError('HAM batch16 preflight not passed')
    with (O/(mode+'.log')).open('x') as log:
     p=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT);state(mode,child_pid=p.pid,gpu=0);update({'preflight':'preflight','train':'training','test':'testing'}[mode],f'SkinFormer HAM {mode}',p.pid)
     if p.wait()!=0:raise RuntimeError(f'{mode} failed; see {O/(mode+".log")}')
   if not read(OUT/'test_complete.json').get('complete'):raise RuntimeError('final tests did not complete')
   if not all(read(OUT/'final_test'/m/'artifact_audit.json').get('complete') for m in ('none','mirror4')):raise RuntimeError('audit missing')
   state('complete');update('complete','HAM 300轮和两种测试及审计完成')
  except BaseException as exc:
   state('failed',error=repr(exc),traceback=traceback.format_exc());update('failed',repr(exc));raise
if __name__=='__main__':main()

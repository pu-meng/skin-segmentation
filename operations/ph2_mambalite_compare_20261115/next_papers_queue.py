import json,os,subprocess,time,traceback
from pathlib import Path
R=Path('/home/PuMengYu/experiments/skin_segmentation');O=R/'operations/ph2_mambalite_compare_20261115';B=Path('/media/8T/PuMengYu/experiments/skin_segmentation');P='/home/PuMengYu/anaconda3/envs/medseg/bin/python'
def read(p):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return {}
def state(s,**k):
 p=O/'next_papers_queue_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(status=s,pid=os.getpid(),time=time.time(),**k),indent=2)+'\n');t.replace(p)
def update(v,s,note):
 subprocess.run([P,str(R/'operations/expctl.py'),'update','--id',f'PH2_{v.upper()}_SEED42_FRESH300_20261115','--status',s,'--gpu','1','--pid',str(os.getpid()),'--note',note],check=True,cwd=R,stdout=subprocess.DEVNULL)
def alive(pid):
 try:os.kill(int(pid),0);return True
 except (OSError,ValueError,TypeError):return False
try:
 for v in ('CMUNeXt','FuseUNet'):update(v,'queued','等待前序训练、两种测试和审计')
 while True:
  c=read(B/'PH2_UNET2D_seed42_20261115/UNet2D/controller_status.json');done=read(B/'PH2_UNET2D_seed42_20261115/UNet2D/completion.json')
  if c.get('status')=='complete' and done.get('complete'):break
  if c.get('status')=='failed' or not alive(c.get('controller_pid')):raise RuntimeError(f'UNet2D failed or controller not live: {c}')
  state('waiting_UNet2D',phase=c.get('status'));time.sleep(20)
 for v in ('CMUNeXt','FuseUNet'):
  op=R/f'operations/ph2_{v.lower()}_20261115'
  with (op/'controller.log').open('x') as log:
   p=subprocess.Popen([P,'-B','-u',str(op/'controller.py'),v],cwd=R,stdout=log,stderr=subprocess.STDOUT)
   while p.poll() is None:state('running',model=v,child_pid=p.pid);time.sleep(20)
   if p.returncode:raise RuntimeError(f'{v} failed {p.returncode}; see {op}/controller.log')
  done=read(B/f'PH2_{v.upper()}_seed42_20261115'/v/'completion.json');assert done.get('complete')
 state('complete')
except BaseException as e:
 state('failed',error=repr(e),traceback=traceback.format_exc())
 for v in ('CMUNeXt','FuseUNet'):
  a=read(B/f'PH2_{v.upper()}_seed42_20261115'/v/'controller_status.json')
  if a.get('status') not in ('complete','failed'):update(v,'stalled',f'PH2接续中断: {e}')
 raise

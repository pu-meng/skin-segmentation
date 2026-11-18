import json,os,subprocess,time,traceback
from pathlib import Path
R=Path('/home/PuMengYu/experiments/skin_segmentation');O=R/'operations/ph2_cmunext_author_20261115';U=Path('/media/8T/PuMengYu/experiments/skin_segmentation/PH2_UNET2D_seed42_20261115/UNet2D');P='/home/PuMengYu/anaconda3/envs/medseg/bin/python'
def read(p):
 try:return json.loads(p.read_text())
 except (ValueError,OSError):return {}
def status(s,**kw):
 p=O/'queue_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(status=s,pid=os.getpid(),time=time.time(),**kw),ensure_ascii=False,indent=2)+'\n');t.replace(p)
def update(s,note):subprocess.run([P,str(R/'operations/expctl.py'),'update','--id','PH2_CMUNEXT_AUTHOR_SEED41_FRESH300_20261115','--status',s,'--gpu','1','--pid',str(os.getpid()),'--note',note],cwd=R,check=True,stdout=subprocess.DEVNULL)
def alive(pid):
 try:os.kill(int(pid),0);return True
 except (OSError,ValueError,TypeError):return False
try:
 update('queued','等待 UNet2D 完成训练、双测试和审计后启动作者配置预检')
 while True:
  c=read(U/'controller_status.json');d=read(U/'completion.json')
  if c.get('status')=='complete' and d.get('complete'):break
  if c.get('status')=='failed' or not alive(c.get('controller_pid')):raise RuntimeError(f'UNet predecessor failed or exited: {c}')
  status('waiting_UNet2D',phase=c.get('status'));time.sleep(15)
 with (O/'controller.log').open('x') as log:
  p=subprocess.Popen([P,'-B','-u',str(O/'controller.py'),'CMUNeXt'],cwd=R,stdout=log,stderr=subprocess.STDOUT)
  while p.poll() is None:status('running',child_pid=p.pid);time.sleep(15)
  if p.returncode:raise RuntimeError(f'CMUNeXt controller failed: {p.returncode}')
 status('complete')
except BaseException as e:
 status('failed',error=repr(e),traceback=traceback.format_exc());update('failed',repr(e));raise

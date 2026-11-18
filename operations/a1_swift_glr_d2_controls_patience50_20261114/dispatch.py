import json,os,subprocess,time,traceback
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation');OPS=ROOT/'operations/a1_swift_glr_d2_controls_patience50_20261114';PREV=ROOT/'operations/a1_swift_glr_d2_c4c5_patience50_20261114';OUT=Path('/media/8T/PuMengYu/experiments/skin_segmentation');DOC=ROOT/'研究记录/实验主线整理_HAM_ISIC.md';PY='/home/PuMengYu/anaconda3/envs/medseg/bin/python'
def read(p):
 try:return json.loads(p.read_text())
 except FileNotFoundError:return {}
def status(s,**kw):
 p=OPS/'queue_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(status=s,pid=os.getpid(),updated_at=time.time(),**kw),ensure_ascii=False,indent=2)+'\n');t.replace(p)
def update(v,s,note):
 subprocess.run([PY,str(ROOT/'operations/expctl.py'),'update','--id',f'A1_SWIFTFORMER_GLR_D2_{v}_ISIC_SEED42_PATIENCE50_20261114','--status',s,'--gpu','0','--pid',str(os.getpid()),'--note',note],check=True,cwd=ROOT,stdout=subprocess.DEVNULL)
def alive(pid):
 try:os.kill(int(pid),0);return True
 except (OSError,TypeError,ValueError):return False
def record(v):
 work=OUT/f'A1_SWIFTFORMER_GLR_D2_{v}_ISIC_SEED42_PATIENCE50_20261114'/'ISIC2018'/f'A1-GLR-D2-{v}';completion=read(work/'completion.json')
 if not completion.get('complete'):raise RuntimeError(f'{v} missing completion')
 vals=[]
 for mode in ('none','mirror4'):
  path=work/'final_test'/mode;summary=read(path/'summary.json');audit=read(path/'artifact_audit.json')
  if summary.get('images')!=518 or not audit.get('complete'):raise RuntimeError(f'{v} {mode} incomplete')
  vals.append(f'[{summary["global"]["dice"]*100:.4f}]({os.path.relpath(path/"summary.json",DOC.parent)})')
 s=DOC.read_text();prefix=f'| A1-SwiftFormer-S-GLR-D2-{v} |';lines=s.splitlines(keepends=True);ids=[i for i,l in enumerate(lines) if l.startswith(prefix)]
 if len(ids)!=1:raise RuntimeError('missing or duplicate row '+v)
 lines[ids[0]]=f'{prefix} {vals[0]} | {vals[1]} | seed42；patience50 {completion["epochs"]}轮；各518张；双模式产物审计通过 |\n'
 tmp=DOC.with_name(DOC.name+f'.{os.getpid()}.tmp');tmp.write_text(''.join(lines));tmp.replace(DOC)
try:
 for v in ('CTRL-U','CTRL-AVG'):update(v,'queued','等待 C5 完成双测试和审计后依次接续')
 while True:
  s=read(PREV/'queue_status.json')
  if s.get('status')=='complete':break
  if s.get('status')=='failed' or not alive(s.get('pid')):raise RuntimeError(f'C4/C5 queue not running or failed: {s}')
  status('waiting_C5',predecessor=s.get('status'));time.sleep(30)
 for v in ('CTRL-U','CTRL-AVG'):
  with (OPS/f'{v}_queue.log').open('x') as log:
   proc=subprocess.Popen([PY,'-B','-u',str(OPS/'controller.py'),v],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
   while proc.poll() is None:status('running',variant=v,child_pid=proc.pid);time.sleep(30)
   if proc.returncode:raise RuntimeError(f'{v} failed {proc.returncode}')
  record(v)
 status('complete')
except BaseException as exc:
 status('failed',error=repr(exc),traceback=traceback.format_exc())
 for v in ('CTRL-U','CTRL-AVG'):
  if read(OPS/f'{v}_status.json').get('status') not in ('complete','failed'):update(v,'stalled',f'前序接续失败：{exc}')
 raise

import json,os,subprocess,time,traceback
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation');OPS=ROOT/'operations/a1_swift_glr_d2_c4c5_20261114';PREV=ROOT/'operations/a1_swift_glr_d2_c2c3_20261114';PY='/home/PuMengYu/anaconda3/envs/medseg/bin/python';OUT=Path('/media/8T/PuMengYu/experiments/skin_segmentation');DOC=ROOT/'研究记录/实验主线整理_HAM_ISIC.md'
def read(p):
 try:return json.loads(p.read_text())
 except FileNotFoundError:return {}
def state(s,**kw):
 p=OPS/'queue_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(status=s,pid=os.getpid(),updated_at=time.time(),**kw),ensure_ascii=False,indent=2)+'\n');t.replace(p)
def update(v,s,note,pid=None):
 args=[PY,str(ROOT/'operations/expctl.py'),'update','--id',f'A1_SWIFTFORMER_GLR_D2_{v}_ISIC_SEED42_FRESH300_20261114','--status',s,'--gpu','0','--note',note]
 if pid:args+=['--pid',str(pid)]
 subprocess.run(args,check=True,cwd=ROOT,stdout=subprocess.DEVNULL)
def alive(pid):
 try:os.kill(int(pid),0);return True
 except (OSError,TypeError,ValueError):return False
def record(v):
 work=OUT/f'A1_SWIFTFORMER_GLR_D2_{v}_ISIC_SEED42_FRESH300_20261114'/'ISIC2018'/f'A1-GLR-D2-{v}'
 if not read(work/'completion.json').get('complete'):return False
 vals=[]
 for mode in ('none','mirror4'):
  target=work/'final_test'/mode;summary=read(target/'summary.json');audit=read(target/'artifact_audit.json')
  if not audit.get('complete') or summary.get('images')!=518:return False
  vals.append((summary['global']['dice']*100,target/'summary.json'))
 s=DOC.read_text();prefix=f'| A1-SwiftFormer-S-GLR-D2-{v} |';lines=s.splitlines(keepends=True);matches=[i for i,l in enumerate(lines) if l.startswith(prefix)]
 if len(matches)!=1:raise RuntimeError(f'missing or duplicate result row {v}')
 links=[]
 for value,path in vals:
  rel=os.path.relpath(path,DOC.parent);links.append(f'[{value:.4f}]({rel})')
 lines[matches[0]]=f'{prefix} {links[0]} | {links[1]} | seed42；各518张；双模式产物审计通过 |\n'
 tmp=DOC.with_name(DOC.name+f'.{os.getpid()}.tmp');tmp.write_text(''.join(lines));tmp.replace(DOC);return True
def wait_for_c3():
 while True:
  for v in ('C2','C3'):record(v)
  s=read(PREV/'C3_status.json')
  if s.get('status')=='complete':
   if not record('C3'):raise RuntimeError('C3 completion artifacts missing')
   return
  if s.get('status')=='failed' or not alive(s.get('controller_pid')):raise RuntimeError(f'C3 did not complete and controller is not alive: {s}')
  state('waiting_C3',predecessor=s.get('status'));time.sleep(30)
try:
 for v in ('C4','C5'):update(v,'queued','C3完成双测试和审计后，按C4→C5顺序接续',os.getpid())
 wait_for_c3()
 for v in ('C4','C5'):
  state('running',variant=v)
  with (OPS/f'{v}_queue.log').open('x') as log:
   proc=subprocess.Popen([PY,'-B','-u',str(OPS/'controller.py'),v],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
   while proc.poll() is None:
    record('C2');state('running',variant=v,child_pid=proc.pid);time.sleep(30)
   if proc.returncode:raise RuntimeError(f'{v} controller failed: {proc.returncode}')
  if not record(v):raise RuntimeError(f'{v} completion artifacts missing')
 record('C2');state('complete')
except BaseException as exc:
 state('failed',error=repr(exc),traceback=traceback.format_exc())
 for v in ('C4','C5'):
  s=read(OPS/f'{v}_status.json')
  if s.get('status') not in ('complete','failed'):update(v,'stalled',f'GLR接续队列中断：{exc}')
 raise

import fcntl,json,os,time,traceback
from pathlib import Path
R=Path('/home/PuMengYu/experiments/skin_segmentation');B=Path('/media/8T/PuMengYu/experiments/skin_segmentation');P=R/'研究记录/老师汇报_表1_TTA结果.md';O=R/'operations/ph2_mambalite_compare_20261115';Q=O/'next_papers_queue_status.json'
def read(p):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return {}
def update(v):
 root=B/f'PH2_{v.upper()}_seed42_20261115'/v;done=read(root/'completion.json')
 if not done.get('complete'):return False
 vals=[]
 for mode in ('none','mirror4'):
  target=root/'final_test'/mode;summary=read(target/'summary.json');audit=read(target/'artifact_audit.json')
  if summary.get('images')!=40 or not audit.get('complete'):return False
  vals.append((summary['global']['dice']*100,os.path.relpath(target/'summary.json',P.parent)))
 with (O/'teacher_report.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);lines=P.read_text().splitlines(keepends=True);prefix=f'| {v} |';ids=[i for i,line in enumerate(lines) if line.startswith(prefix)];assert len(ids)==1
  lines[ids[0]]=f'{prefix} [{vals[0][0]:.4f}]({vals[0][1]}) | [{vals[1][0]:.4f}]({vals[1][1]}) | {done["best_epoch"]} | 两模式各40张，产物审计通过 |\n';t=P.with_name(P.name+f'.{os.getpid()}.tmp');t.write_text(''.join(lines));t.replace(P)
 return True
try:
 remaining={'UNet2D','CMUNeXt','FuseUNet'}
 while remaining:
  for v in tuple(remaining):
   if update(v):remaining.remove(v)
  if not remaining:break
  q=read(Q)
  if q.get('status')=='failed':raise RuntimeError('PH2 paper queue failed: '+str(q.get('error')))
  pid=q.get('pid')
  try:os.kill(int(pid),0)
  except (OSError,ValueError,TypeError):raise RuntimeError('PH2 queue controller not live')
  time.sleep(30)
 (O/'report_sync_status.json').write_text(json.dumps({'status':'complete','models':['UNet2D','CMUNeXt','FuseUNet']})+'\n')
except BaseException as e:
 (O/'report_sync_status.json').write_text(json.dumps({'status':'failed','error':repr(e),'traceback':traceback.format_exc()},ensure_ascii=False,indent=2)+'\n');raise

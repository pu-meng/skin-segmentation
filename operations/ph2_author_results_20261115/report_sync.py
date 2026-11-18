import fcntl,json,os,time,traceback
from pathlib import Path
R=Path('/home/PuMengYu/experiments/skin_segmentation');E=Path('/media/8T/PuMengYu/experiments/skin_segmentation');OPS=R/'operations';DOCS=[R/'研究记录/实验主线整理_HAM_ISIC.md',R/'研究记录/老师汇报_表1_TTA结果.md'];F=OPS/'ph2_author_results_20261115';jobs={'FocalTransNet':('ph2_focal_author_20261115','PH2_FOCALTRANSNET_AUTHOR_seed1234_20261115','final_test_audit/none','final_test_audit/mirror4'),'AD-GBC':('ph2_adgbc_author_20261115','PH2_ADGBC_AUTHOR_seed1029_20261115','final_test/ours','final_test/ours_tta'),'S3H-Net':('ph2_s3hnet_author_20261115','PH2_S3HNET_AUTHOR_seed3407_20261115','final_test/none','final_test/mirror4'),'FuseUNet':('ph2_fuseunet_author_20261115','PH2_FUSEUNET_AUTHOR_seed42_20261115','final_test/none','final_test/mirror4')}
def read(p):return json.loads(p.read_text())
def atomic(p,s):t=p.with_name(p.name+f'.{os.getpid()}.tmp');t.write_text(s);t.replace(p)
def update(name,job):
 _,base,a,b=job;dirs=[E/base/a,E/base/b];s=[read(p/'summary.json') for p in dirs];aud=[read(p/'artifact_audit.json') for p in dirs];assert all(x['complete'] for x in aud) and all(x['images']==40 for x in s)
 for doc in DOCS:
  vals=[f'[{x["global"]["dice"]*100:.4f}]({os.path.relpath(p/"summary.json",doc.parent)})' for x,p in zip(s,dirs)];lines=doc.read_text().splitlines(keepends=True)
  if doc==DOCS[0]:prefix=f'| PH2-{name}（作者配置） |';ids=[i for i,v in enumerate(lines) if v.startswith(prefix)];new=f'{prefix} {vals[0]} | {vals[1]} | 两模式各40张；审计通过 |\n'
  else:
   start=next(i for i,v in enumerate(lines) if '### 表1c' in v);end=next((i for i in range(start+1,len(lines)) if lines[i].startswith('### ')),len(lines));ids=[i for i in range(start,end) if lines[i].startswith(f'| {name} |')];epoch=s[0].get('selected_epoch',s[0].get('best_epoch','—'));new=f'| {name} | {vals[0]} | {vals[1]} | {epoch} | 作者配置；两模式各40张，审计通过 |\n'
  assert len(ids)==1,(name,doc,ids);lines[ids[0]]=new;atomic(doc,''.join(lines))
finished=set()
while len(finished)<len(jobs):
 for name,job in jobs.items():
  if name in finished:continue
  try:
   status=read(OPS/job[0]/'queue_status.json')
   if status['status']=='complete':
    with (F/'report.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX);update(name,job)
    finished.add(name);print('RECORDED',name,flush=True)
   elif status['status']=='failed':
    finished.add(name);print('QUEUE_FAILED',name,status.get('error'),flush=True)
  except Exception as e:
   print('SYNC_WAIT_OR_ERROR',name,repr(e),flush=True)
 time.sleep(20)

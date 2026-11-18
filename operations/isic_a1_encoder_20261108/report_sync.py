"""Fill the ISIC A1 encoder table when each full train/test audit is complete."""
import csv,json,time,fcntl
from pathlib import Path

RUN=Path('/media/8T/PuMengYu/experiments/skin_segmentation/ISIC2018_A1_EncoderSwap_FRESH300_20261108')
DOC=Path('/home/PuMengYu/论文汇总/四层分割实验总览/实验结果汇总.md')
NAMES={'rmt_t':'RMT-T','rmt_s':'RMT-S','swiftformer_s':'SwiftFormer-S','mobileone_s1':'MobileOne-S1'}
BASE='../../../../media/8T/PuMengYu/experiments/skin_segmentation/ISIC2018_A1_EncoderSwap_FRESH300_20261108'

def read(p):return json.loads(Path(p).read_text())
def row(arm):
 work=RUN/arm;c=read(work/'completion.json')
 assert c.get('complete') is True and c.get('epochs')==300 and c.get('test_images_per_mode')==518
 audit=read(work/'run/artifact_audit.json');assert audit.get('training_complete') and audit.get('epochs')==300
 for mode in ['none','mirror4']:assert read(work/f'final_test/{mode}/artifact_audit.json').get('complete') is True
 no=read(work/'final_test/none/summary.json');tta=read(work/'final_test/mirror4/summary.json')
 assert no['dataset']==tta['dataset']=='ISIC2018' and no['split']=='test' and tta['split']=='test'
 with (work/'run/history.csv').open(newline='') as f:hist=list(csv.DictReader(f))
 assert len(hist)==300 and all(float(x['seconds'])>0 for x in hist)
 sec=sum(float(x['seconds']) for x in hist)/len(hist)
 npath=f'{BASE}/{arm}/final_test/none/summary.json';tpath=f'{BASE}/{arm}/final_test/mirror4/summary.json'
 return f"|ISIC2018|A1＋{NAMES[arm]}|[{no['global']['dice']*100:.4f}]({npath})|[{tta['global']['dice']*100:.4f}]({tpath})|{sec:.1f}|seed42 fresh300；测试518张；[运行记录]({BASE}/{arm}/completion.json)|"

def sync_once():
 with DOC.with_suffix('.md.sync.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);text=DOC.read_text();changed=[]
  for arm,name in NAMES.items():
   complete=RUN/arm/'completion.json'
   if not complete.exists():continue
   new=row(arm);prefix=f'|ISIC2018|A1＋{name}|';lines=text.splitlines();idx=next((i for i,s in enumerate(lines) if s.startswith(prefix)),None)
   if idx is not None and lines[idx]!=new:lines[idx]=new;text='\n'.join(lines)+'\n';changed.append(arm)
  if changed:
   tmp=DOC.with_suffix('.md.sync.tmp');tmp.write_text(text);tmp.replace(DOC)
  return changed

if __name__=='__main__':
 while True:
  try:
   changed=sync_once()
   if changed:print('UPDATED',','.join(changed),flush=True)
   if all((RUN/a/'completion.json').exists() for a in NAMES):break
  except Exception as e:print('SYNC_RETRY',repr(e),flush=True)
  time.sleep(60)

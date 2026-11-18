"""Update the ISIC early-comparison TTA column from audited artifacts."""
import csv,fcntl,json,time
from pathlib import Path

ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
RUN=ROOT/'log/ISIC2018_瓶颈对比_20261031/mirror4_TTA_20261108'
DOC=Path('/home/PuMengYu/论文汇总/四层分割实验总览/实验结果汇总.md')
DISPLAY={
 'tinyvim_base':'TinyViM基线','tinyvim_v1':'TinyViM v1','tinyvim_v2':'TinyViM v2',
 'tinyvim_v2_ds_proj':'TinyViM v2＋DS投影','tinyvim_v2_no_proj':'TinyViM v2无投影','tinyvim_v3':'TinyViM v3',
 'unet_base':'UNet基线','unet_v1':'UNet v1','unet_v2':'UNet v2',
 'unet_v2_ds_proj':'UNet v2＋DS投影','unet_v2_no_proj':'UNet v2无投影','unet_v3':'UNet v3'
}

def read(p):return json.loads(Path(p).read_text())

def sync_once():
 with DOC.with_suffix('.md.ttasync.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);text=DOC.read_text();changed=[]
  for arm,name in DISPLAY.items():
   status=RUN/arm/'status.json';audit=RUN/arm/'mirror4/artifact_audit.json';summary=RUN/arm/'mirror4/summary.json'
   if not(status.is_file() and audit.is_file() and summary.is_file()):continue
   st=read(status);au=read(audit);res=read(summary)
   if st.get('status')!='complete' or not au.get('complete') or au.get('baseline_no_tta_exact_mask_matches')!=518:continue
   assert res['dataset']=='ISIC2018' and res['tta']=='mirror4' and res['images']==518
   value=f"[{res['global']['dice']*100:.4f}]({RUN}/{arm}/mirror4/summary.json)"
   lines=text.splitlines();prefix=f'|ISIC2018|{name}|';idx=next((i for i,s in enumerate(lines) if s.startswith(prefix)),None)
   if idx is None:continue
   cells=lines[idx].split('|')
   # Table layout: dataset, method, no-TTA, TTA, seconds/epoch.
   assert len(cells)==7 and cells[3].strip() not in ('补测中','待补测') or len(cells)==7
   if cells[4].strip()!=value:
    cells[4]=value;lines[idx]='|'.join(cells);text='\n'.join(lines)+'\n';changed.append(arm)
  if changed:
   tmp=DOC.with_suffix('.md.ttasync.tmp');tmp.write_text(text);tmp.replace(DOC)
  return changed

if __name__=='__main__':
 while True:
  try:
   changed=sync_once()
   if changed:print('UPDATED_TTA',','.join(changed),flush=True)
   if all((RUN/a/'status.json').is_file() and read(RUN/a/'status.json').get('status')=='complete' for a in DISPLAY):break
  except Exception as e:print('SYNC_RETRY',repr(e),flush=True)
  time.sleep(30)

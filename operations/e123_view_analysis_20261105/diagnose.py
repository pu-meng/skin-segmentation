import os,sys,json,time,csv,fcntl,traceback,hashlib
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
OUT=ROOT/'operations/e123_view_analysis_20261105'
sys.path.insert(0,str(ROOT/'operations/e123_early_mirror4_20261105'))
import run_test as e
import numpy as np
import torch
from PIL import Image

def main():
 torch.set_num_threads(2);e.api.seed_all(42);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.25,0)
 assert e.sha(e.CP)==e.EXPECTED['E123.1']
 protected=e.identities();e.write(OUT/'protected_before.json',protected)
 data=e.api.HAMDataset(Path(e.PATHS['active_split']).parent,'test');assert len(data)==2020
 assert len({r['image_id'] for r in data.rows})==2020
 (OUT/'views').mkdir(exist_ok=True)
 e.write(OUT/'protocol.json',dict(checkpoint=str(e.CP),sha256=e.sha(e.CP),selected_epoch=215,images=2020,views=[[],[2],[3],[2,3]],fusion='inverse flip real logits mean then sigmoid>=0.5',precision='FP32',batch=1,workers=0,boundary='GT 4-neighbor boundary distance <=3 pixels; >3 inside and outside; fixed before analysis',case_selection='two largest Dice gains, two largest losses, two nearest median delta; unique IDs tie lexicographic',size_groups='GT foreground fraction <0.10, [0.10,0.30), >=0.30; descriptive fixed',no_view_or_threshold_selection=True))
 model=e.build('E123.1');model.load_state_dict(torch.load(e.CP,map_location='cpu'),strict=True);model.cuda().eval()
 pred=e.Predictor(model,True).eval();records=[];start=time.time()
 with torch.no_grad():
  for i,r in enumerate(data.rows):
   ident=r['image_id'];x=data[i][0].unsqueeze(0).cuda();logits=[]
   for axes in [(),(2,),(3,),(2,3)]:
    l=pred.logits(torch.flip(x,axes) if axes else x)
    logits.append(torch.flip(l,axes) if axes else l)
   # Match mirror_core's left-to-right FP32 accumulation exactly.
   avg=((logits[0]+logits[1])+logits[2])+logits[3];avg=avg/4
   fused=(torch.sigmoid(avg)[0,0]>=.5).cpu().numpy()
   ls=torch.cat(logits,dim=0)[:,0];probs=torch.sigmoid(ls)
   masks=(probs>=.5).cpu().numpy()
   old=np.asarray(Image.open(e.OLD['E123.1']/'predictions'/(ident+'_segmentation.png')))>0
   tta=np.asarray(Image.open(e.OUT/'test/predictions'/(ident+'_segmentation.png')))>0
   a=int(np.count_nonzero(masks[0]!=old));b=int(np.count_nonzero(fused!=tta))
   assert a==b==0,(ident,a,b)
   variance=probs.var(0,unbiased=False)
   record=dict(image_id=ident,original_mismatch_pixels=a,fused_mismatch_pixels=b,prob_variance_mean=float(variance.mean()),prob_variance_max=float(variance.max()),elapsed_seconds=time.time()-start)
   np.savez_compressed(OUT/'views'/(ident+'.npz'),packed=np.packbits(masks.reshape(4,-1),axis=1))
   records.append(record)
   if (i+1)%100==0 or i==0 or i+1==2020:
    e.write(OUT/'status.json',dict(status='diagnostic_inference',completed=i+1,total=2020,pid=os.getpid(),seconds=time.time()-start));print('DIAGNOSTIC',i+1,2020,round(time.time()-start,2),flush=True)
 assert e.identities()==protected
 e.write(OUT/'protected_after.json',e.identities());e.write(OUT/'view_inference.json',dict(complete=True,images=len(records),pixelwise_original_and_fused_match_all=True,records=records))
 pred.close();e.write(OUT/'status.json',dict(status='inference_complete',completed=2020,pid=os.getpid(),seconds=time.time()-start))
if __name__=='__main__':
 try:
  with (OUT/'diagnostic.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main()
 except BaseException:
  e.write(OUT/'status.json',dict(status='failed',traceback=traceback.format_exc()));raise

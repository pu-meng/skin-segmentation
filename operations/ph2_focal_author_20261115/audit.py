#!/usr/bin/env python3
"""Retest the fixed FocalTransNet ISIC checkpoint; preserve original output."""
import csv, importlib.util, json, os, sys, time
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
OPS=ROOT/'operations/focaltransnet_isic_audit_20261114'
RUN=Path('/media/8T/PuMengYu/experiments/skin_segmentation/PH2_FOCALTRANSNET_AUTHOR_seed1234_20261115')
OUT=RUN/'final_test_audit'
spec=importlib.util.spec_from_file_location('focal_isic_fixed',ROOT/'operations/focaltransnet_isic_20261114/run.py')
focal=importlib.util.module_from_spec(spec);spec.loader.exec_module(focal)
focal.CACHE=ROOT/'operations/ph2_focal_author_20261115/cache';focal.SPLIT=ROOT/'data/processed/PH2_256_ratio_7_1_2_seed42_20261115/split.csv'
def write(p,v):
 p=Path(p);tmp=p.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def main():
 if OUT.exists():raise FileExistsError(f'保留已有输出，拒绝覆盖: {OUT}')
 old=json.loads((RUN/'final_test.json').read_text());checkpoint=RUN/'best.pt';digest=focal.sha256(checkpoint)
 if digest!=old['checkpoint_sha256']:raise RuntimeError('fixed checkpoint hash differs')
 state=torch.load(checkpoint,map_location='cpu',weights_only=False);model=focal.build_model();model.load_state_dict(state['state_dict'],strict=True)
 if torch.cuda.is_initialized():raise RuntimeError('CUDA initialized before strict CPU load')
 ds=focal.IsicSplit('test',False);ids=[r['image_id'] for r in ds.rows]
 if len(ids)!=40 or len(set(ids))!=40:raise RuntimeError('unexpected test IDs')
 loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=0);OUT.mkdir();model.cuda().eval()
 write(OUT/'cpu_load_audit.json',{'strict':True,'checkpoint_sha256':digest,'selected_epoch':state['epoch']})
 for mode in ('none','mirror4'):
  dest=OUT/mode; pred_dir=dest/'predictions';pred_dir.mkdir(parents=True);vis=dest/'visualizations';vis.mkdir()
  seen=[]; rows=[];totals=np.zeros(4,dtype=np.int64)
  with torch.inference_mode():
   for x,y,image_id in loader:
    x=x.cuda();truth=y.cuda().bool();logits=model(x)
    if mode=='mirror4':logits=(logits+model(x.flip(-1)).flip(-1)+model(x.flip(-2)).flip(-2)+model(x.flip((-2,-1))).flip((-2,-1)))/4
    prediction=logits.argmax(1).bool(); counts=[int((prediction&truth).sum()),int((prediction&~truth).sum()),int((~prediction&truth).sum()),int((~prediction&~truth).sum())]
    totals+=counts;tp,fp,fn,tn=counts;name=image_id[0];seen.append(name)
    rows.append(dict(image_id=name,tp=tp,fp=fp,fn=fn,tn=tn,dice=2*tp/max(2*tp+fp+fn,1)))
    image=(prediction[0].cpu().numpy().astype(np.uint8)*255)
    if not cv2.imwrite(str(pred_dir/f'{name}.png'),image):raise RuntimeError('prediction write failed')
    if len(seen)<=12:
     rgb=((x[0].cpu().numpy().transpose(1,2,0)*.5+.5)*255).clip(0,255).astype(np.uint8)
     combined=np.concatenate((cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),cv2.cvtColor(y[0].numpy().astype(np.uint8)*255,cv2.COLOR_GRAY2BGR),cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)),axis=1)
     if not cv2.imwrite(str(vis/f'{name}.png'),combined):raise RuntimeError('visualization write failed')
    if len(seen)%100==0:print(mode,len(seen),flush=True)
  if seen!=ids:raise RuntimeError('test ID sequence changed')
  with (dest/'per_image_metrics.csv').open('w',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
  tp,fp,fn,tn=map(int,totals);orig=old['results'][mode]
  matches=all(int(orig[k])==v for k,v in [('tp',tp),('fp',fp),('fn',fn)])
  if not matches:raise RuntimeError(f'{mode} counts differ from original: {orig} vs {tp,fp,fn}')
  summary={'dataset':'PH2','model':'FocalTransNet','split':'test','images':40,'selected_epoch':int(state['epoch']),
   'checkpoint_sha256':digest,'split_sha256':focal.sha256(focal.SPLIT),'tta':mode=='mirror4','input_size':224,
   'global':{'dice':2*tp/max(2*tp+fp+fn,1),'iou':tp/max(tp+fp+fn,1),'accuracy':(tp+tn)/max(tp+tn+fp+fn,1),
             'sensitivity':tp/max(tp+fn,1),'specificity':tn/max(tn+fp,1),'tp':tp,'fp':fp,'fn':fn,'tn':tn},
   'test_used_for_checkpoint_selection':False,'original_counts_match':matches}
  (dest/'report.txt').write_text('Fixed checkpoint retest, 40 images; Global Dice and IoU from pooled pixels.\n')
  audit={'complete':True,'prediction_count':len(list(pred_dir.glob('*.png'))),'metric_count':len(rows),
         'visualizations':len(list(vis.glob('*.png'))),'ids_match_frozen_test_split':True,
         'checkpoint_sha256':digest,'original_tp_fp_fn_match':matches}
  if audit['prediction_count']!=40 or audit['visualizations']!=12:raise RuntimeError('artifact count mismatch')
  write(dest/'summary.json',summary);write(dest/'artifact_audit.json',audit);print('AUDITED',mode,summary['global'],flush=True)
 write(OUT/'completion.json',{'complete':True,'checkpoint_sha256':digest,'images_per_mode':40,'original_counts_match':True,'completed_at':time.time()})
if __name__=='__main__':main()

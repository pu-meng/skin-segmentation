import csv,json,time
from pathlib import Path
import cv2,numpy as np
from scipy import ndimage
R=Path('/home/PuMengYu/experiments/skin_segmentation');E=Path('/media/8T/PuMengYu/experiments/skin_segmentation');D=R/'data/processed/ham10000_group42';O=R/'operations/ham_completed_metrics_20261115';masks=np.load(D/'masks.npy',mmap_mode='r')
with (D/'split.csv').open() as f:rows={x['image_id']:int(x['cache_index']) for x in csv.DictReader(f) if x['split']=='test'}
assert len(rows)==2020
for model,folder,size in [('FocalTransNet','FocalTransNet_HAM_FRESH100_20261114',224),('S3H-Net','S3HNet_HAM_FRESH200_20261114',256)]:
 out=O/('hd95_'+model.replace('-','_')+'_HAM_mirror4.json')
 if out.exists():
  print('ALREADY_DONE',model,flush=True);continue
 d=E/folder/'final_test/mirror4'; audit=json.loads((d/'artifact_audit.json').read_text());assert audit['complete'] and audit['prediction_count']==2020
 with (d/'per_image_metrics.csv').open() as f:cases=list(csv.DictReader(f))
 assert len(cases)==2020 and {x['image_id'] for x in cases}==set(rows);values=[];empty=0;start=time.time()
 for row in cases:
  key=row['image_id'];raw=np.asarray(masks[rows[key]]);y=(raw>0 if model=='FocalTransNet' else raw>=128).astype(np.uint8)
  if size==224:y=cv2.resize(y,(224,224),interpolation=cv2.INTER_NEAREST)
  y=y.astype(bool);p=np.load(d/'predictions'/(key+'.npy')).squeeze().astype(bool);assert p.shape==y.shape
  cnt=[int((p&y).sum()),int((p&~y).sum()),int((~p&y).sum()),int((~p&~y).sum())];assert cnt==[int(row[k]) for k in ('tp','fp','fn','tn')],key
  if not p.any() or not y.any():v=0. if not p.any() and not y.any() else float(np.hypot(*p.shape));empty+=1
  else:
   sp=p^ndimage.binary_erosion(p);sy=y^ndimage.binary_erosion(y);dist=np.concatenate((ndimage.distance_transform_edt(~sy)[sp],ndimage.distance_transform_edt(~sp)[sy]));v=float(np.percentile(dist,95,method='linear'))
  values.append(v)
 result={'model':model,'dataset':'HAM10000','tta':'mirror4','count':len(values),'mean_hd95_pixels':float(np.mean(values)),'empty_cases':empty,'input_size':size,'all_per_image_confusion_counts_matched':True,'source_summary':str(d/'summary.json'),'seconds':round(time.time()-start,1),'method':'mean of per-image 95th percentile symmetric surface distances; empty mismatch=image diagonal'}
 path=O/('hd95_'+model.replace('-','_')+'_HAM_mirror4.json');path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

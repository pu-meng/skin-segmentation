import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv,json,hashlib,math,time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
R=Path('/home/PuMengYu/experiments/skin_segmentation')
RUN=R/'log/TinyViM_E123.1_联合新版_HAM_20261105'
OUT=RUN/'early_test/current_best_20261105'
W=OUT/'workers0/E123.1';T=W/'test'
for _ in range(300):
 if (W/'artifact_audit.json').exists():break
 time.sleep(2)
else:raise TimeoutError('early test incomplete')
s=json.loads((T/'summary.json').read_text());fixed=json.loads((OUT/'snapshot.json').read_text());paths=json.loads((R/'source/config/paths.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
split=Path(paths['active_split']);assert sha(split)==fixed['split_sha256']
with split.open() as f:allrows=list(csv.DictReader(f))
rows=[r for r in allrows if r['split']=='test'];ids={r['image_id'] for r in rows};assert len(rows)==len(ids)==2020
assert not ids & {r['image_id'] for r in allrows if r['split']!='test'}
with (T/'per_image_metrics.csv').open() as f:mr=list(csv.DictReader(f))
assert len(mr)==2020 and {r['image_id'] for r in mr}==ids
byid={r['image_id']:r for r in mr}
assert {p.name for p in (T/'predictions').glob('*.png')}=={i+'_segmentation.png' for i in ids}
masks=np.load(Path(paths['cache_root'])/'masks.npy',mmap_mode='r');total=np.zeros(4,dtype=np.int64);ds=[];hs=[]
for r in rows:
 p=np.asarray(Image.open(T/'predictions'/f'{r["image_id"]}_segmentation.png'));assert p.shape==(256,256) and set(np.unique(p)).issubset({0,255})
 p=p>0;g=masks[int(r['cache_index'])]>0
 c=np.array([(p&g).sum(),(p&~g).sum(),(~p&g).sum(),(~p&~g).sum()]);total+=c
 m=byid[r['image_id']]
 for k,v in zip(['tp','fp','fn','tn'],c):assert int(m[k])==int(v),(r['image_id'],k)
 tp,fp,fn,tn=c;d=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.;ds.append(d);assert abs(d-float(m['dice']))<1e-12
 if not p.any() or not g.any():h=0. if not p.any() and not g.any() else math.hypot(*p.shape)
 else:
  a=p^ndimage.binary_erosion(p);b=g^ndimage.binary_erosion(g)
  h=float(np.percentile(np.concatenate([ndimage.distance_transform_edt(~b)[a],ndimage.distance_transform_edt(~a)[b]]),95,method='linear'))
 assert abs(h-float(m['hd95_pixels']))<1e-10;hs.append(h)
for p in (T/'visualizations').glob('*.png'):
 with Image.open(p) as im:im.load();assert im.size==(768,256)
assert len(list((T/'visualizations').glob('*.png')))==12
for k,v in zip(['tp','fp','fn','tn'],total):assert s['global'][k]==int(v)
assert abs(np.mean(ds)-s['per_image_mean']['dice'])<1e-12
assert abs(np.mean(hs)-s['per_image_mean']['hd95_pixels'])<1e-12
assert sha(fixed['checkpoint'])==fixed['checkpoint_sha256']
e0path=R/'log/TinyViM_四层对应Decoder_基础_HAM_20261104/最终测试/four_stage/test/summary.json'
e0=json.loads(e0path.read_text());assert e0['split_sha256']==s['split_sha256']
result=dict(complete=True,pixel_confusion_and_hd95_recomputed_images=2020,predictions=2020,csv_rows=2020,visualizations_decoded=12,checkpoint_sha256=fixed['checkpoint_sha256'],selected_epoch=s['selected_epoch'],global_metrics=s['global'],per_image_mean=s['per_image_mean'],e0_summary=str(e0path),e0_global=e0['global'],delta_vs_E0_pp={k:100*(s['global'][k]-e0['global'][k]) for k in ['dice','iou','accuracy','sensitivity','specificity']},delta_hd95_pixels=s['per_image_mean']['hd95_pixels']-e0['per_image_mean']['hd95_pixels'],source_best_sha_now=sha(RUN/'run/checkpoints/best.pth'),training_status_now=json.loads((RUN/'run/status.json').read_text()),test_elapsed_seconds=s['elapsed_seconds'])
(W/'independent_audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
print(json.dumps({k:v for k,v in result.items() if k!='training_status_now'},ensure_ascii=False,indent=2))
print('TRAINING_EPOCH_NOW',result['training_status_now'].get('epoch_completed'))

import json,csv,hashlib,math,time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
RUN=Path('/media/8T/PuMengYu/experiments/skin_segmentation/DecoderDualA_SCREEN100of300_20261106')
OUT=RUN/'screen100_best39_test_20261107'
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def hd(p,g):
 if not p.any() or not g.any():return 0. if not p.any() and not g.any() else math.hypot(*p.shape)
 sp=p^ndimage.binary_erosion(p);sg=g^ndimage.binary_erosion(g)
 return float(np.percentile(np.concatenate([ndimage.distance_transform_edt(~sg)[sp],ndimage.distance_transform_edt(~sp)[sg]]),95,method='linear'))
completion=json.loads((OUT/'completion.json').read_text());assert completion['complete']
m=json.loads((RUN/'protocol.json').read_text());s=json.loads((OUT/'DualA/test/summary.json').read_text());snapshot=json.loads((OUT/'snapshot.json').read_text())
for name,h in m['source_sha256'].items():assert sha(ROOT/name)==h
assert sha(RUN/'protocol.json')==snapshot['protocol_sha256']
assert sha(snapshot['checkpoint'])==sha(snapshot['source_checkpoint'])==s['checkpoint_sha256']
split=Path(m['config']['data']['split_file']);assert sha(split)==s['split_sha256']
rows=[r for r in csv.DictReader(split.open()) if r['split']=='test'];ids={r['image_id'] for r in rows};assert len(rows)==len(ids)==2020
output=OUT/'DualA/test';metrics=list(csv.DictReader((output/'per_image_metrics.csv').open()));lookup={r['image_id']:r for r in metrics};assert len(metrics)==len(lookup)==2020 and set(lookup)==ids
assert {p.name for p in (output/'predictions').glob('*.png')}=={i+'_segmentation.png' for i in ids}
masks=np.load(m['config']['data']['cache_files']['masks.npy']['real_path'],mmap_mode='r');counts=np.zeros(4,dtype=np.int64);dist=[]
for row in rows:
 p=np.asarray(Image.open(output/'predictions'/f'{row["image_id"]}_segmentation.png'))>0;g=masks[int(row['cache_index'])]>0
 assert p.shape==g.shape==(256,256)
 c=[int((p&g).sum()),int((p&~g).sum()),int((~p&g).sum()),int((~p&~g).sum())];r=lookup[row['image_id']]
 assert c==[int(r[k]) for k in ['tp','fp','fn','tn']]
 d=hd(p,g);assert abs(d-float(r['hd95_pixels']))<1e-12;counts+=c;dist.append(d)
for k,v in zip(['tp','fp','fn','tn'],counts):assert int(v)==s['global'][k]
assert abs(float(np.mean(dist))-s['per_image_mean']['hd95_pixels'])<1e-12
visuals=list((output/'visualizations').glob('*.png'));assert len(visuals)==12
for p in visuals:
 with Image.open(p) as im:im.verify()
assert (output/'report.txt').is_file()
record=dict(complete=True,independent_png_vs_fixed_gt_pixel_audit=True,images=2020,unique_ids=2020,predictions=2020,metrics=2020,visualizations=12,all_per_image_confusion_equal=True,all_per_image_hd95_equal=True,checkpoint_sha256=s['checkpoint_sha256'],training_sources_unchanged=True,global_metrics=s['global'],mean_hd95_pixels=float(np.mean(dist)),full_pipeline_elapsed_seconds=completion['full_pipeline_elapsed_seconds'],source_sha256=m['source_sha256'])
(OUT/'independent_artifact_audit.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(record,ensure_ascii=False))

import csv,hashlib,io,json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation'); SRC=ROOT/'data/raw/PH2_hf_ahishamm_20261115/ph2_masks.parquet'; OUT=ROOT/'data/processed/PH2_256_ratio_7_1_2_seed42_20261115'
assert not OUT.exists(),f'refuse overwrite {OUT}'
def sha(b):return hashlib.sha256(b).hexdigest()
table=pq.read_table(SRC).to_pylist();assert len(table)==200
records=[];seen=set();image_hashes=set()
for row in table:
 x,y=row['image'],row['label'];id=Path(x['path']).stem
 assert id==Path(y['path']).stem and id not in seen and id.startswith('IMD');seen.add(id)
 im=Image.open(io.BytesIO(x['bytes']));mask=Image.open(io.BytesIO(y['bytes']));im.load();mask.load()
 assert im.mode=='RGB' and im.size==mask.size==(256,256)
 a=np.asarray(mask.convert('L'));assert set(np.unique(a)).issubset({0,255}) and (a>0).any() and (a==0).any()
 ih=sha(im.tobytes());assert ih not in image_hashes;image_hashes.add(ih)
 records.append({'image_id':id,'image_bytes':x['bytes'],'mask_bytes':y['bytes'],'image_sha256':sha(x['bytes']),'mask_sha256':sha(y['bytes'])})
records.sort(key=lambda r:r['image_id']);rng=np.random.default_rng(42);permutation=rng.permutation(200);assignment={int(index):('train' if rank<140 else 'val' if rank<160 else 'test') for rank,index in enumerate(permutation)}
OUT.mkdir(parents=True);(OUT/'images').mkdir();(OUT/'masks').mkdir();rows=[]
for i,r in enumerate(records):
 for sub,key in [('images','image_bytes'),('masks','mask_bytes')]: (OUT/sub/(r['image_id']+'.png')).write_bytes(r[key])
 rows.append({k:r[k] for k in ('image_id','image_sha256','mask_sha256')}|{'split':assignment[i]})
with (OUT/'split.csv').open('x',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['image_id','split','image_sha256','mask_sha256']);w.writeheader();w.writerows(rows)
counts={s:sum(r['split']==s for r in rows) for s in ('train','val','test')};assert counts=={'train':140,'val':20,'test':40}
manifest={'source':'https://huggingface.co/datasets/ahishamm/ph2_masks','source_revision':'b08479ce4a8e9b4526b7ce97c422131912b985db','source_parquet_sha256':sha(SRC.read_bytes()),'source_caveat':'Community mirror provides paired 256x256 PNG; preprocessing and mask provenance not documented; not claimed identical to official PH2 originals.','split_method':'sorted IMD IDs; numpy default_rng(42).permutation(200); 140 train, 20 validation, 40 test','split_sha256':sha((OUT/'split.csv').read_bytes()),'paper_ratio':'7:1:2; exact paper split IDs not disclosed and not claimed reproduced','counts':counts,'image_and_mask_pairs':200,'image_pixel_exact_duplicates':0,'patient_level_independence':'not verified'}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n');print(json.dumps(manifest,ensure_ascii=False))

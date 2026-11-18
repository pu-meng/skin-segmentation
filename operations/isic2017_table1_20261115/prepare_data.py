import csv, hashlib, io, json, zipfile
from pathlib import Path
from PIL import Image
import numpy as np
src=Path('/media/8T/PuMengYu/system_disk_archive/SSD_ISIC2017迁移_20261023/ISIC2017')
out=Path('/media/8T/PuMengYu/experiments/skin_segmentation/ISIC2017_official_256_20261115')
assert not out.exists(), f'output exists: {out}'
files={'train':('ISIC-2017_Training_Data.zip','ISIC-2017_Training_Part1_GroundTruth.zip',2000),'val':('ISIC-2017_Validation_Data.zip','ISIC-2017_Validation_Part1_GroundTruth.zip',150),'test':('ISIC-2017_Test_v2_Data.zip','ISIC-2017_Test_v2_Part1_GroundTruth.zip',600)}
out.mkdir(parents=True);(out/'images').mkdir();(out/'masks').mkdir();rows=[];seen=set()
for split,(images,masks,n) in files.items():
 with zipfile.ZipFile(src/images) as zi, zipfile.ZipFile(src/masks) as zm:
  ix={Path(p).stem:p for p in zi.namelist() if p.lower().endswith('.jpg')};mx={Path(p).stem.removesuffix('_segmentation'):p for p in zm.namelist() if p.endswith('_segmentation.png')}
  assert len(ix)==len(mx)==n and set(ix)==set(mx) and not seen.intersection(ix); seen.update(ix)
  for i,key in enumerate(sorted(ix)):
   im=Image.open(io.BytesIO(zi.read(ix[key]))).convert('RGB');mask=Image.open(io.BytesIO(zm.read(mx[key]))).convert('L');assert im.size==mask.size
   im=im.resize((256,256),Image.Resampling.BILINEAR);mask=mask.resize((256,256),Image.Resampling.NEAREST)
   assert set(np.unique(mask)).issubset({0,255})
   paths=[out/'images'/f'{key}.png',out/'masks'/f'{key}.png']; im.save(paths[0]);mask.save(paths[1]); hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
   rows.append(dict(image_id=key,split=split,image_sha256=hashes[0],mask_sha256=hashes[1]))
   if (i+1)%200==0: print(split,i+1,flush=True)
with (out/'split.csv').open('x',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['image_id','split','image_sha256','mask_sha256']);w.writeheader();w.writerows(rows)
(out/'manifest.json').write_text(json.dumps({'source':'ISIC 2017 official challenge archives','split':'official challenge train/validation/test','counts':{s:sum(r['split']==s for r in rows) for s in files},'image_resize':'bilinear 256x256 RGB','mask_resize':'nearest 256x256 binary','patient_independence':'not verified','paper_split_identity':'not verified','split_sha256':hashlib.sha256((out/'split.csv').read_bytes()).hexdigest()},indent=2)+'\n')
print('DONE',len(rows),flush=True)

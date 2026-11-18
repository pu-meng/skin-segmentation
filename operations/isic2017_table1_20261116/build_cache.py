import csv,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
R=Path('/home/PuMengYu/experiments/skin_segmentation');D=R/'data/processed/ISIC2017_去重_seed42_20261116';O=R/'operations/isic2017_table1_20261116/cache';O.mkdir(exist_ok=False)
rows=list(csv.DictReader((D/'split.csv').open()));n=len(rows);img=np.lib.format.open_memmap(O/'images.npy',mode='w+',dtype=np.uint8,shape=(n,256,256,3));mask=np.lib.format.open_memmap(O/'masks.npy',mode='w+',dtype=np.uint8,shape=(n,256,256))
for i,row in enumerate(rows):
 for sub,col in [('images','image_sha256'),('masks','mask_sha256')]:
  p=D/sub/(row['image_id']+'.png');assert hashlib.sha256(p.read_bytes()).hexdigest()==row[col],str(p)
 img[i]=np.asarray(Image.open(D/'images'/(row['image_id']+'.png')).convert('RGB'));mask[i]=np.asarray(Image.open(D/'masks'/(row['image_id']+'.png')).convert('L'));row['cache_index']=i;row['dx']='all'
 if (i+1)%500==0:print('CACHE',i+1,'/',n,flush=True)
img.flush();mask.flush()
with (O/'split.csv').open('x',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(O/'cache_complete.json').write_text(json.dumps({'dedup_split_sha256':hashlib.sha256((D/'split.csv').read_bytes()).hexdigest(),'images':n,'file_hashes_checked':True},indent=2)+'\n');print('CACHE_COMPLETE',n,flush=True)

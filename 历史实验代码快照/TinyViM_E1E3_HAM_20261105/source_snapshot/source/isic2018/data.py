"""ISIC-only cache and datasets; the HAM loaders and config remain unchanged."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import csv,hashlib,io,json,os,tempfile,zipfile
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
DEDUP=ROOT/'data/processed/isic2018_dedup_seed42_20261031'
CACHE=ROOT/'data/processed/isic2018_256_dedup_seed42_20261031'
ZIP_CACHE={}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return list(csv.DictReader(Path(p).open()))
def decode(r):
 raw=[]
 for prefix in ['image','mask']:
  p=r[prefix+'_archive']
  if p not in ZIP_CACHE:ZIP_CACHE[p]=zipfile.ZipFile(p)
  b=ZIP_CACHE[p].read(r[prefix+'_member']);assert hashlib.sha256(b).hexdigest()==r[prefix+'_sha256'];raw.append(b)
 with Image.open(io.BytesIO(raw[0])) as im:x=np.asarray(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
 with Image.open(io.BytesIO(raw[1])) as im:y=(np.asarray(im.convert('L').resize((256,256),Image.Resampling.NEAREST))>0).astype(np.uint8)*255
 assert y.any(),r['image_id']
 return x,y

def prepare():
 if CACHE.exists():
  verify();print('CACHE_ALREADY_VERIFIED',str(CACHE),flush=True);return
 rows=read(DEDUP/'split.csv');inventory={r['image_id']:r for r in read(DEDUP/'source_inventory.csv')}
 assert len(rows)==2556 and len({r['image_id'] for r in rows})==2556
 m=json.loads((DEDUP/'manifest.json').read_text());assert sha(DEDUP/'split.csv')==m['split_sha256']
 tmp=Path(tempfile.mkdtemp(prefix=CACHE.name+'.staging.',dir=CACHE.parent))
 images=np.lib.format.open_memmap(tmp/'images.npy',mode='w+',dtype=np.uint8,shape=(len(rows),256,256,3))
 masks=np.lib.format.open_memmap(tmp/'masks.npy',mode='w+',dtype=np.uint8,shape=(len(rows),256,256))
 with ProcessPoolExecutor(max_workers=4) as pool:
  for i,(x,y) in enumerate(pool.map(decode,[inventory[r['image_id']] for r in rows],chunksize=4)):
   images[i]=x;masks[i]=y
   if (i+1)%250==0:print('CACHE',i+1,'/',len(rows),flush=True)
 images.flush();masks.flush();del images,masks
 enriched=[{**r,'cache_index':i,'dx':'all'} for i,r in enumerate(rows)]
 with (tmp/'split.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(enriched[0]));w.writeheader();w.writerows(enriched)
 manifest={'dataset':'ISIC2018','images':len(rows),'counts':m['counts'],'dedup_split_file':str(DEDUP/'split.csv'),'dedup_split_sha256':sha(DEDUP/'split.csv'),'split_sha256':sha(tmp/'split.csv'),'files':{n:{'sha256':sha(tmp/n),'bytes':(tmp/n).stat().st_size} for n in ['images.npy','masks.npy']},'preprocessing':{'image_resize':'PIL RGB bilinear to256','mask_resize':'PIL L nearest then >0, store0/255','normalization':'RGB float64 /255 then float32; encoder-local ImageNet normalization for TinyViM','exif':'stored pixel orientation, matching parent preparation','category':'all means no diagnostic category supplied; no inferred clinical labels'},'all_splits_preprocessed':True,'test_used_for_model_selection':False}
 (tmp/'cache_complete.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');os.rename(tmp,CACHE);verify();print('CACHE_COMPLETE',str(CACHE),flush=True)

def verify():
 m=json.loads((CACHE/'cache_complete.json').read_text());assert sha(DEDUP/'split.csv')==m['dedup_split_sha256'];assert sha(CACHE/'split.csv')==m['split_sha256']
 for n,info in m['files'].items():assert sha(CACHE/n)==info['sha256'],n
 rows=read(CACHE/'split.csv');assert [(r['image_id'],r['split']) for r in rows]==[(r['image_id'],r['split']) for r in read(DEDUP/'split.csv')]
 x=np.load(CACHE/'images.npy',mmap_mode='r');y=np.load(CACHE/'masks.npy',mmap_mode='r');assert x.shape==(2556,256,256,3) and y.shape==(2556,256,256)
 assert x.dtype==y.dtype==np.uint8
 return m

import sys
sys.path.insert(0,str(ROOT/'source/comparison'))
from ham_fast_pipeline import PlannedDataset,AugmentationPlanSampler,EpochLoader,worker_init
class ISICDataset(PlannedDataset):
 def __init__(self,data,split):
  self.data=Path(data);self.cache=CACHE;self.training=split=='train';self.rows=[r for r in read(self.data/'split.csv') if r['split']==split]
  self.images=np.load(self.cache/'images.npy',mmap_mode='r');self.masks=np.load(self.cache/'masks.npy',mmap_mode='r')
 def __setstate__(self,state):
  self.__dict__.update(state);self.images=np.load(self.cache/'images.npy',mmap_mode='r');self.masks=np.load(self.cache/'masks.npy',mmap_mode='r')

def loaders(config,api):
 import torch
 from torch.utils.data import DataLoader
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(42);reference=api.MambaLiteUNet();state=torch.get_rng_state();del reference
 generator=torch.Generator();generator.set_state(state);result={}
 for split in ['train','val']:
  dataset=ISICDataset(CACHE,split);sampler=AugmentationPlanSampler(api.ReferenceRandomSampler(dataset,generator)) if split=='train' else None
  loader=DataLoader(dataset,batch_size=8,sampler=sampler,num_workers=2,pin_memory=True,drop_last=False,generator=torch.Generator().manual_seed(42001),worker_init_fn=worker_init,persistent_workers=True,prefetch_factor=2,multiprocessing_context='spawn')
  result[split]=EpochLoader(loader,generator)
 result['val_precise']=DataLoader(result['val'].dataset,batch_size=1,num_workers=2,pin_memory=True,generator=torch.Generator().manual_seed(42002),worker_init_fn=worker_init,persistent_workers=True,multiprocessing_context='spawn')
 assert len(result['train'])==223
 return result,generator

if __name__=='__main__':prepare()

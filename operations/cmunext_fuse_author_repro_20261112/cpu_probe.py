import os,sys,json,hashlib
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import pandas as pd
import torch
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
sys.path.insert(0,str(ROOT/'source/lightweight_repro_20261111'))
from models import build_model
from dataset import SkinSplitDataset
from train_repro import DATASETS
result={'kind':'CPU synthetic model, split metadata, and first train sample probe; no training or test image read','models':{},'datasets':{}}
torch.set_num_threads(1)
for name in ('cmunext','fuseunet'):
 m=build_model(name); x=torch.rand(1,3,64,64); y=m(x); y.mean().backward()
 result['models'][name]={'shape':list(y.shape),'finite_output':bool(torch.isfinite(y).all()),'finite_gradients':all(bool(torch.isfinite(p.grad).all()) for p in m.parameters() if p.grad is not None),'parameter_count':sum(p.numel() for p in m.parameters())}
for name,c in DATASETS.items():
 frame=pd.read_csv(c['split_csv']); ids={s:set(frame.loc[frame['split']==s,'image_id'].astype(str)) for s in ('train','val','test')}; counts={s:len(frame.loc[frame['split']==s]) for s in ids}
 assert counts==c['counts'] and all(len(ids[s])==counts[s] for s in ids)
 assert not(ids['train']&ids['val'] or ids['train']&ids['test'] or ids['val']&ids['test'])
 d=SkinSplitDataset(c['images_npy'],c['masks_npy'],c['split_csv'],'train',size=256,train_mode=False);x,y,_=d[0]
 result['datasets'][name]={'counts':counts,'unique_disjoint_ids':True,'split_sha256':hashlib.sha256(Path(c['split_csv']).read_bytes()).hexdigest(),'first_train_shape':list(x.shape),'first_train_min':float(x.min()),'first_train_max':float(x.max()),'first_train_mean':float(x.mean()),'mask_values':torch.unique(y).tolist()}
result['cuda_initialized']=torch.cuda.is_initialized()
assert not result['cuda_initialized']
out=Path('/home/PuMengYu/experiments/skin_segmentation/operations/cmunext_fuse_author_repro_20261112/cpu_probe.json')
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

import os, sys, json, csv, time, random, hashlib, traceback
from pathlib import Path
import numpy as np, pandas as pd, cv2, torch
from torch.utils.data import Dataset, DataLoader
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
sys.path.insert(0,str(ROOT/'references/code/CMUNeXt'))
from network.CMUNeXt import cmunext
from utils.losses import BCEDiceLoss
from utils.metrics import iou_score
P=ROOT/'data/processed'
CONFIG={'HAM10000':(P/'ham10000_group42/images.npy',P/'ham10000_group42/masks.npy',P/'ham10000_group42_dedup_20261023/split.csv'), 'ISIC2018':(P/'isic2018_256_dedup_seed42_20261031/images.npy',P/'isic2018_256_dedup_seed42_20261031/masks.npy',P/'isic2018_256_dedup_seed42_20261031/split.csv')}
class Data(Dataset):
 def __init__(self, cfg, split):
  self.images=np.load(cfg[0],mmap_mode='r'); self.masks=np.load(cfg[1],mmap_mode='r'); f=pd.read_csv(cfg[2]); f=f[f['split']==split]; self.indices=f.cache_index.astype(int).tolist(); self.ids=f.image_id.astype(str).tolist(); self.train=split=='train'
 def __len__(self): return len(self.indices)
 def __getitem__(self,i):
  x=np.asarray(self.images[self.indices[i]])[:,:,::-1].copy(); y=np.asarray(self.masks[self.indices[i]]).copy()
  if self.train:
   if random.random()<0.5:
    k=random.randint(0,3); x=np.rot90(x,k); y=np.rot90(y,k)
   if random.random()<0.5:
    code=random.choice([-1,0,1]); x=cv2.flip(np.ascontiguousarray(x),code); y=cv2.flip(np.ascontiguousarray(y),code)
  x=cv2.resize(np.ascontiguousarray(x),(256,256),interpolation=cv2.INTER_LINEAR); y=cv2.resize(np.ascontiguousarray(y),(256,256),interpolation=cv2.INTER_NEAREST)
  x=(x.astype(np.float32)/255.0-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
  x=x/255.0; y=y.astype(np.float32)/255.0
  return torch.from_numpy(x.transpose(2,0,1).copy()),torch.from_numpy(y[None].copy()),self.ids[i]
def main():
 ds=sys.argv[1]; out=Path(sys.argv[2]); out.mkdir(parents=True,exist_ok=True); cfg=CONFIG[ds]
 torch.manual_seed(41); np.random.seed(41); random.seed(41); torch.cuda.manual_seed_all(41); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
 protocol={'dataset':ds,'physical_gpu':0,'seed':41,'epochs':300,'batch_size':8,'split_csv':str(cfg[2]),'split_sha256':hashlib.sha256(cfg[2].read_bytes()).hexdigest(),'source':'references/code/CMUNeXt','loss':'official BCEDiceLoss','optimizer':'SGD lr=.01 momentum=.9 wd=.0001; post-step poly .9','selection':'max mean per-image author IoU on val, strict >; initial 0','test':'only after 300 epochs; fixed validation best','deviations':['project cached 256x256 images and masks + project split instead of author file layout; RGB cache converted to BGR; cached masks retained (HAM contains intermediate values)','albumentations unavailable: manually implement RandomRotate90 p=.5, Flip p=.5 with three directions, resize, Normalize; RNG implementation differs','literal author Normalize then divide 255 retained: normalized input is divided by 255 a second time; unusual magnitude, not corrected','test uses threshold .5 and global pixel confusion summary; no TTA'],'gpu_sharing':'explicitly authorized; no batch reduction on OOM'}
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2,ensure_ascii=False))
 train=DataLoader(Data(cfg,'train'),batch_size=8,shuffle=True,num_workers=8,pin_memory=False,drop_last=False); val=DataLoader(Data(cfg,'val'),batch_size=1,shuffle=False,num_workers=1)
 model=cmunext().cuda(); lossfn=BCEDiceLoss().cuda(); optim=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9,weight_decay=.0001); maxit=len(train)*300; it=0; best=0.0
 print('PID',os.getpid(),'CUDA_VISIBLE_DEVICES',os.environ.get('CUDA_VISIBLE_DEVICES'),'train',len(train.dataset),'val',len(val.dataset),'steps',len(train),flush=True)
 with (out/'history.csv').open('w',newline='',buffering=1) as f:
  w=csv.writer(f); w.writerow(['epoch','train_loss','val_loss','val_iou','seconds'])
  for epoch in range(300):
   start=time.time(); model.train(); total=0.; n=0
   for x,y,_ in train:
    x=x.cuda(); y=y.cuda(); z=model(x); loss=lossfn(z,y); optim.zero_grad(); loss.backward(); optim.step()
    for pg in optim.param_groups: pg['lr']=.01*(1-it/maxit)**.9
    it+=1; total+=loss.item()*len(x); n+=len(x)
    if it==1: print('FIRST_OPTIMIZER_STEP', 'pid',os.getpid(),'batch',len(x),'loss',loss.item(),'cuda',torch.cuda.get_device_name(0),flush=True)
   model.eval(); vs=0.; vi=0.; count=0
   with torch.no_grad():
    for x,y,_ in val:
     x=x.cuda(); y=y.cuda(); z=model(x); vs+=lossfn(z,y).item(); vi+=iou_score(z,y)[0]; count+=1
   vi/=count; vs/=count; w.writerow([epoch+1,total/n,vs,vi,time.time()-start]); print('EPOCH',epoch+1,'train_loss',total/n,'val_loss',vs,'val_iou',vi,flush=True)
   if vi>best:
    best=vi; torch.save(model.state_dict(),out/'best.pth'); (out/'selection.json').write_text(json.dumps({'epoch':epoch+1,'val_iou':vi})); print('BEST',epoch+1,vi,flush=True)
 if not (out/'best.pth').exists(): raise RuntimeError('no best: validation IoU never exceeded 0')
 model.load_state_dict(torch.load(out/'best.pth',map_location='cuda:0')); model.eval(); test=DataLoader(Data(cfg,'test'),batch_size=1,shuffle=False,num_workers=1); tp=fp=fn=tn=0
 with torch.no_grad(), (out/'test_per_image.csv').open('w',newline='') as f:
  w=csv.writer(f); w.writerow(['id','tp','fp','fn','tn'])
  for x,y,ids in test:
   p=(torch.sigmoid(model(x.cuda()))>0.5).cpu(); t=y>0.5
   a=int((p&t).sum()); b=int((p&~t).sum()); c=int((~p&t).sum()); d=int((~p&~t).sum()); tp+=a; fp+=b; fn+=c; tn+=d; w.writerow([ids[0],a,b,c,d])
 summary={'split':'test','images':len(test.dataset),'checkpoint':'best.pth','selection':json.loads((out/'selection.json').read_text()),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'global_dice':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.0,'global_iou':tp/(tp+fp+fn) if tp+fp+fn else 1.0}; (out/'test_summary.json').write_text(json.dumps(summary,indent=2)); print('COMPLETE',json.dumps(summary),flush=True)
if __name__=='__main__':
 try: main()
 except BaseException:
  traceback.print_exc(); raise

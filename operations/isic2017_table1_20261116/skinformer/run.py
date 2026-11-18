#!/usr/bin/env python3
"""Local SkinFormer adaptation, fixed HAM split; train/validate/test separated."""
import argparse,csv,hashlib,json,os,random,sys,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation');OPS=ROOT/'operations/isic2017_table1_20261116/skinformer';VENDOR=ROOT/'operations/skinformer_adapt_20261114/vendor';HAM=ROOT/'data/processed/ISIC2017_去重_seed42_20261116'
sys.path[:0]=[str(VENDOR),str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
import numpy as np,torch
from torch.utils.data import DataLoader
from Models.networks.net import skinformer
from project_paths import CACHE as HAM_CACHE
from final_test_pipeline import validate_outputs
from pretrained_unet2d import prepare_imports
prepare_imports("mambavision_t")
import baseline_runner_seed42 as api

def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def seed():random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
def model():return skinformer(SimpleNamespace(out_size=256),n_classes=2)
class Probability(torch.nn.Module):
 def __init__(self,network,tta=False):super().__init__();self.network=network;self.tta=tta
 def forward(self,x):
  p=self.network(x)[:,1:2]
  if self.tta:
   for dims in ((2,),(3,),(2,3)):p=p+self.network(x.flip(dims))[:,1:2].flip(dims)
   p=p/4
  return p
class Dice(torch.nn.Module):
 def forward(self,p,y):return 1-(2*(p*y).sum()+1e-6)/((p.square()+y.square()).sum()+1e-6)
class PH2(torch.utils.data.Dataset):
 def __init__(self,split):
  with (HAM/'split.csv').open(newline='') as f:self.rows=[r for r in csv.DictReader(f) if r['split']==split]
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  from PIL import Image
  r=self.rows[i];x=np.asarray(Image.open(HAM/'images'/f"{r['image_id']}.png").convert('RGB'),dtype=np.float32)/255.;y=np.asarray(Image.open(HAM/'masks'/f"{r['image_id']}.png").convert('L'),dtype=np.uint8)>0
  return torch.from_numpy(np.ascontiguousarray(x)).permute(2,0,1),torch.from_numpy(np.ascontiguousarray(y)).unsqueeze(0).float(),r['image_id'],'all'
def loader(split,batch,shuffle=False):
 ds=PH2(split);expected={'train':1997,'val':149,'test':600}[split];assert len(ds)==len({r['image_id'] for r in ds.rows})==expected
 return DataLoader(ds,batch_size=batch,shuffle=shuffle,num_workers=0,pin_memory=True),ds
def manifest():
 files=sorted((VENDOR/'Models').rglob('*.py'))+[OPS/'run.py',OPS/'preflight.py',ROOT/'source/comparison/baseline_runner_seed42.py',HAM/'split.csv']
 return {str(f):sha(f) for f in files}
def train(out):
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);seed();torch.set_num_threads(4);torch.cuda.set_device(0)
 net=model().cuda();pnet=Probability(net);criterion=Dice();opt=torch.optim.Adam(net.parameters(),lr=.0002,weight_decay=1e-8);scheduler=torch.optim.lr_scheduler.StepLR(opt,step_size=256,gamma=.5)
 dl,_=loader('train',16,True);vl,ds=loader('val',1);assert len(ds)==149
 write(out/'protocol.json',{'model':'SkinFormer local adaptation; missing layers reconstructed, author identity unverified','dataset':'ISIC2017_deduplicated','seed':42,'batch':16,'epochs':300,'optimizer':'Adam lr=0.0002 weight_decay=1e-8','scheduler':'StepLR 256 gamma 0.5','loss':'squared-denominator foreground Dice, epsilon 1e-6','augmentation':'independent horizontal and vertical flips p=0.5','input':'RGB/255 256x256','selection':'max validation pooled global Dice, earliest tie','test_used_for_selection':False,'source_sha256':manifest()})
 ck=out/'checkpoints';ck.mkdir();best=-1;best_epoch=0
 for epoch in range(1,301):
  net.train();start=time.monotonic();losses=[]
  for x,y,_,_ in dl:
   x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True)
   if random.random()<.5:x=x.flip(-1);y=y.flip(-1)
   if random.random()<.5:x=x.flip(-2);y=y.flip(-2)
   opt.zero_grad(set_to_none=True);loss=criterion(pnet(x),y)
   if not bool(torch.isfinite(loss)):raise RuntimeError('nonfinite training loss')
   loss.backward();opt.step();losses.append(float(loss.detach()))
   if epoch==1 and len(losses)==1:write(out/'first_optimizer_update.json',{'passed':True,'pid':os.getpid(),'batch':int(x.shape[0]),'gpu':os.getenv('EXPERIMENT_PHYSICAL_GPU')})
  scheduler.step();val=api.evaluate(pnet,vl,criterion,torch.device('cuda:0'));score=val['global']['dice']
  if not np.isfinite(score):raise RuntimeError('nonfinite validation score')
  if score>best:best=score;best_epoch=epoch;torch.save(net.state_dict(),ck/'best.pth')
  tmp=ck/'latest.tmp';torch.save({'epoch':epoch,'model':net.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'best':best,'best_epoch':best_epoch,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state()},tmp);tmp.replace(ck/'latest.pth')
  row={'epoch':epoch,'train_loss':float(np.mean(losses)),'val_loss':val['loss'],'val_global_dice':score,'best_epoch':best_epoch,'seconds':time.monotonic()-start}
  with (out/'history.csv').open('a',newline='') as f:
   w=csv.DictWriter(f,fieldnames=row.keys());
   if epoch==1:w.writeheader()
   w.writerow(row)
  write(out/'status.json',{'status':'training','pid':os.getpid(),'epoch_completed':epoch,'epochs':300,**row});print(json.dumps(row),flush=True)
 write(out/'train_complete.json',{'complete':True,'epochs':300,'best_epoch':best_epoch,'best_val_global_dice':best,'checkpoint_sha256':sha(ck/'best.pth')})
def test(out):
 done=json.loads((out/'train_complete.json').read_text());assert done['complete'] and done['epochs']==300
 p=json.loads((out/'protocol.json').read_text());assert p['source_sha256']==manifest(),'source drift';cp=out/'checkpoints/best.pth';assert sha(cp)==done['checkpoint_sha256'];seed();torch.set_num_threads(4)
 net=model();net.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True),strict=True);assert not torch.cuda.is_initialized();net.cuda().eval();dl,ds=loader('test',1);ids={r['image_id'] for r in ds.rows};assert len(ids)==len(ds)==600
 result={}
 for mode in ('none','mirror4'):
  target=out/'final_test'/mode
  if target.exists():raise FileExistsError(target)
  target.mkdir(parents=True);r=api.evaluate(Probability(net,mode=='mirror4'),dl,Dice(),torch.device('cuda:0'),target)
  r.update(dataset='ISIC2017_deduplicated',model='SkinFormer local adaptation',seed=42,selected_epoch=done['best_epoch'],checkpoint_sha256=sha(cp),split='test',tta=mode=='mirror4',tta_aggregation='mean foreground probability',test_used_for_checkpoint_selection=False)
  audit=validate_outputs(target,ids,r);assert audit['complete'];write(target/'summary.json',r);write(target/'artifact_audit.json',audit);result[mode]=r
 write(out/'test_complete.json',{'complete':True,'modes':['none','mirror4'],'test_images_per_mode':600,'checkpoint_sha256':sha(cp),'results':result})
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['train','test']);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();(train if a.mode=='train' else test)(a.out)

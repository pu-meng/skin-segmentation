import csv,hashlib,json,os,random,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
import torch
from torch.utils.data import Dataset,DataLoader
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation');DATA=ROOT/'data/processed/PH2_256_ratio_7_1_2_seed42_20261115';OUT=Path('/media/8T/PuMengYu/experiments/skin_segmentation/PH2_UNET2D_seed42_20261115');MODEL=sys.argv[2];PHASE=sys.argv[1];assert MODEL == 'UNet2D'
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
import ham_fast_runner
from final_test_pipeline import validate_outputs
api=ham_fast_runner.baseline_api()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(8388608),b''):h.update(c)
 return h.hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');t.replace(p)
class PH2(Dataset):
 def __init__(self,split):
  self.split=split
  with (DATA/'split.csv').open() as f:self.rows=[r for r in csv.DictReader(f) if r['split']==split]
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i];x=np.asarray(Image.open(DATA/'images'/f'{r["image_id"]}.png').convert('RGB'),dtype=np.float32)/255.;y=np.asarray(Image.open(DATA/'masks'/f'{r["image_id"]}.png').convert('L'),dtype=np.uint8)>0
  if self.split=='train':
   if random.random()>.5:
    k=np.random.randint(0,4);axis=np.random.randint(0,2);x=np.flip(np.rot90(x,k),axis=axis).copy();y=np.flip(np.rot90(y,k),axis=axis).copy()
   if random.random()>.5:
    angle=np.random.randint(20,80);x=ndimage.rotate(x,angle,order=0,reshape=False);y=ndimage.rotate(y,angle,order=0,reshape=False)
  return torch.from_numpy(np.ascontiguousarray(x)).permute(2,0,1).float(),torch.from_numpy(np.ascontiguousarray(y)).unsqueeze(0).float(),r['image_id'],'all'
def build(pretrained=True):
 from source.comparison.unet2d import UNet2D
 return UNet2D()

def seed():api.seed_all(42);torch.set_num_threads(4)
def verify():
 assert sha(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip(),'frozen protocol changed'
 protocol=json.loads((OUT/'protocol.json').read_text());assert sha(DATA/'split.csv')==protocol['split_sha256']
 with (DATA/'split.csv').open() as f:
  for row in csv.DictReader(f):
   for folder,col in (('images','image_sha256'),('masks','mask_sha256')):assert sha(DATA/folder/(row['image_id']+'.png'))==row[col],row['image_id']
 for p,h in protocol['source_sha256'].items():assert sha(p)==h,'source drift: '+p
class Predictor(torch.nn.Module):
 def __init__(self,m,tta):
  super().__init__();self.m=m;self.tta=tta;self.raw=None;self.checked=0
  self.h=m.head.register_forward_hook(self.capture)
 def capture(self,m,i,o):self.raw=o
 def logits(self,x):
  self.raw=None;p=self.m(x);assert self.raw is not None
  z=torch.nn.functional.interpolate(self.raw,size=(256,256),mode='bilinear',align_corners=False)
  if self.checked<3:torch.testing.assert_close(z.sigmoid(),p,rtol=0,atol=0);self.checked+=1
  return z
 def forward(self,x):
  z=self.logits(x)
  if self.tta:
   for dims in ((2,),(3,),(2,3)):z=z+torch.flip(self.logits(torch.flip(x,dims)),dims)
   z=z/4
  return z.sigmoid()
 def close(self):self.h.remove()
def prepare():
 verify();seed();m=build();p=OUT/MODEL;p.mkdir(parents=True,exist_ok=False);torch.save(m.state_dict(),p/'initial.pth')
 write(p/'cpu_acceptance.json',{'passed':True,'model':MODEL,'parameters':sum(z.numel() for z in m.parameters()),'initial_sha256':sha(p/'initial.pth'),'split_sha256':sha(DATA/'split.csv'),'cuda_initialized':torch.cuda.is_initialized(),'initialization':'UNet2D random initialization seed42'})
 print('CPU_PREPARED',MODEL,flush=True)
def run():
 verify();seed();p=OUT/MODEL;torch.cuda.set_device(0);m=build();assert sha(p/'initial.pth')==json.loads((p/'cpu_acceptance.json').read_text())['initial_sha256'];m.load_state_dict(torch.load(p/'initial.pth',map_location='cpu',weights_only=False),strict=True);seed();m.cuda();loss=api.BceDiceLoss().cuda();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8)
 train=DataLoader(PH2('train'),batch_size=8,shuffle=True,num_workers=0,drop_last=False);val=DataLoader(PH2('val'),batch_size=1,shuffle=False,num_workers=0);assert len(train)==18 and len(val.dataset)==20
 if PHASE=='preflight':
  torch.cuda.reset_peak_memory_stats();m.train();x,y,ids,_=next(iter(train));x=x.cuda();y=y.cuda();pred=m(x);l=loss(pred,y);assert pred.shape==(8,1,256,256) and torch.isfinite(l);l.backward();grads=[q.grad for q in m.parameters() if q.grad is not None];assert grads and all(torch.isfinite(q).all() for q in grads);param=next(q for q in m.parameters() if q.grad is not None and q.grad.abs().max()>0);before=param.detach().clone();opt.step();assert (before-param).abs().max()>0
  m.eval();x=next(iter(val))[0].cuda()
  with torch.no_grad():
   for tta in (False,True):
    pp=Predictor(m,tta);z=pp(x);assert z.shape==(1,1,256,256) and torch.isfinite(z).all();pp.close()
  write(p/'preflight.json',{'passed':True,'batch':8,'finite_loss':float(l),'finite_gradients':True,'optimizer_update':True,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'test_data_accessed':False,'gpu':os.getenv('EXPERIMENT_PHYSICAL_GPU')});print('REAL_PREFLIGHT_PASS',MODEL,flush=True);return
 assert PHASE=='train' and json.loads((p/'preflight.json').read_text())['passed']; assert not (p/'history.csv').exists();sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=50,eta_min=.00001);best=float('inf');be=0
 for epoch in range(1,301):
  started=time.time();m.train();losses=[]
  for step,(x,y,ids,_) in enumerate(train,1):
   opt.zero_grad(set_to_none=True);z=m(x.cuda());l=loss(z,y.cuda());assert torch.isfinite(l);l.backward();opt.step();losses.append(float(l.detach()))
   if epoch==1 and step==1:write(p/'first_optimizer_update.json',{'passed':True,'pid':os.getpid(),'step':step,'batch':len(ids),'gpu':os.getenv('EXPERIMENT_PHYSICAL_GPU')});print('FIRST_OPTIMIZER_UPDATE',MODEL,flush=True)
  sched.step();v=api.evaluate(m,val,loss,torch.device('cuda:0'));assert v['images']==20
  if v['loss']<best:best=v['loss'];be=epoch;torch.save(m.state_dict(),p/'best.pth')
  row={'epoch':epoch,'train_loss':float(np.mean(losses)),'val_loss':v['loss'],'val_global_dice':v['global']['dice'],'best_epoch':be,'seconds':time.time()-started}
  with (p/'history.csv').open('a',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(row));
   if epoch==1:w.writeheader()
   w.writerow(row)
  write(p/'status.json',{'status':'training','epoch_completed':epoch,'best_epoch':be,'pid':os.getpid(),'test_data_accessed':False});print('EPOCH',json.dumps(row),flush=True)
 write(p/'training_complete.json',{'epochs':300,'best_epoch':be,'best_validation_loss':best,'best_sha256':sha(p/'best.pth'),'split_sha256':sha(DATA/'split.csv'),'test_data_accessed':False});print('TRAINING_COMPLETE',flush=True)
def test():
 verify();seed();p=OUT/MODEL;done=json.loads((p/'training_complete.json').read_text());assert done['epochs']==300 and done['best_sha256']==sha(p/'best.pth')
 m=build(False);m.load_state_dict(torch.load(p/'best.pth',map_location='cpu',weights_only=False),strict=True);m.cuda().eval();test=PH2('test');ids={r['image_id'] for r in test.rows};assert len(ids)==40;dl=DataLoader(test,batch_size=1,shuffle=False,num_workers=0);crit=api.BceDiceLoss().cuda();results={}
 for mode in ('none','mirror4'):
  folder=p/'final_test'/mode;folder.mkdir(parents=True,exist_ok=False);pred=Predictor(m,mode=='mirror4').eval();result=api.evaluate(pred,dl,crit,torch.device('cuda:0'),folder);pred.close();assert result['images']==40
  result.update(dataset='PH2',model=MODEL,seed=42,selected_epoch=done['best_epoch'],checkpoint_sha256=done['best_sha256'],split_sha256=done['split_sha256'],test_used_for_checkpoint_selection=False,tta=mode=='mirror4');audit=validate_outputs(folder,ids,result);assert audit['complete'];write(folder/'summary.json',result);write(folder/'artifact_audit.json',audit);results[mode]=result;print('TEST_COMPLETE',MODEL,mode,result['global']['dice'],flush=True)
 write(p/'completion.json',{'complete':True,'epochs':300,'test_images_per_mode':40,'best_epoch':done['best_epoch'],'results':results})
if __name__=='__main__':
 if PHASE=='prepare':prepare()
 elif PHASE in ('preflight','train'):run()
 elif PHASE=='test':test()
 else:raise ValueError(PHASE)

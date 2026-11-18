import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import run as m
import torch
from torch.utils.data import DataLoader
from monai.losses import DiceCELoss
from src.models.S3HNet.net import S3HNet
m.pl.seed_everything(m.CFG['seed'],workers=True);torch.set_num_threads(4)
d=m.CachedISIC('train',True);assert len(d)==140
model=S3HNet(**m.CFG['model']).cuda().train();opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.0001);x=next(iter(DataLoader(d,batch_size=8,shuffle=True,num_workers=0)));out=model(x['image'].cuda());y=x['mask'].cuda();crit=DiceCELoss(sigmoid=True,lambda_dice=1,lambda_ce=1);parts=out if isinstance(out,list) else [out];loss=sum(w*crit(z,y) for w,z in zip([1,.5,.25,.125],parts));assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None);opt.step();print('PREFLIGHT_PASS',float(loss.detach()),torch.cuda.max_memory_reserved()/2**20,flush=True)

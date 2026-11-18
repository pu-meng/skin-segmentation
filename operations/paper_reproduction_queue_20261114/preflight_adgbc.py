import sys,json
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
sys.path.insert(0,str(ROOT/'source/adgbc_repro_20261110'))
import torch
import train_adgbc as t

torch.set_num_threads(4)
t.seed_torch(1029)
device=torch.device('cuda:0')
torch.cuda.set_device(device)
model=t.archs_GBC.GBC_Rolling_Unet_L(num_classes=1,input_channels=3,deep_supervision=False,gbc_num_balls=32,gbc_proj_dim=None,use_diag_cov=True,tau=1.0).to(device).train()
criterion=t.losses.BCEDiceWithGeometryLoss(div_weight=.01,scale_weight=.01).to(device)
other,gbc=[],[]
for name,p in model.named_parameters():
 if p.requires_grad:(gbc if 'centers' in name.lower() or 'gbc' in name.lower() else other).append(p)
opt=torch.optim.Adam([{'params':other,'lr':1e-4,'weight_decay':1e-4},{'params':gbc,'lr':.01,'weight_decay':1e-4}])
loader=t.build_loaders(t.DATASETS['HAM10000'],'train',8,0,True)
x,y,_=next(iter(loader));assert tuple(x.shape)==(8,3,256,256)
torch.cuda.reset_peak_memory_stats();opt.zero_grad(set_to_none=True);out=model(x.to(device));loss=criterion(out,y.to(device),model);loss.backward()
assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
opt.step();torch.cuda.synchronize()
Path(sys.argv[1]).write_text(json.dumps({'passed':True,'batch_size':8,'loss':float(loss.detach()),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'dataset':'HAM10000'},indent=2)+'\n')

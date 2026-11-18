"""CPU acceptance of local reconstruction, using validation data only."""
import sys,io,json
from pathlib import Path
from types import SimpleNamespace
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation');OPS=ROOT/'operations/skinformer_adapt_20261114'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(OPS/'vendor'))
import torch
from Models.networks.net import skinformer
from source.isic2018.data import ISICDataset,CACHE

torch.set_num_threads(4);torch.manual_seed(42)
data=ISICDataset(CACHE,'val');x,y,*_=data[0];x=x.unsqueeze(0);y=y.unsqueeze(0)
model=skinformer(SimpleNamespace(out_size=256),n_classes=2).train();opt=torch.optim.Adam(model.parameters(),lr=.0002,weight_decay=1e-8)
pred=model(x);assert pred.shape==(1,2,256,256) and torch.isfinite(pred).all();assert torch.allclose(pred.sum(1),torch.ones_like(pred[:,0]),atol=1e-5)
foreground=pred[:,1:2]; y=y.reshape_as(foreground);loss=1-(2*(foreground*y).sum()+1e-6)/((foreground.square()+y.square()).sum()+1e-6)
loss.backward();grads=[p.grad for p in model.parameters() if p.grad is not None];assert grads and all(torch.isfinite(g).all() for g in grads);opt.step()
buf=io.BytesIO();torch.save(model.state_dict(),buf);buf.seek(0);model.load_state_dict(torch.load(buf,map_location='cpu',weights_only=True),strict=True)
report={'passed':True,'device':'cpu','input':'one ISIC validation sample','shape':list(pred.shape),'output':'two-class softmax','train_loss':float(loss.detach()),'parameters':sum(p.numel() for p in model.parameters()),'parameters_with_gradient':len(grads),'total_parameter_tensors':sum(1 for _ in model.parameters()),'strict_checkpoint_roundtrip':True,'warning':'encoder conv_block and UpCat are locally reconstructed, not verified author implementation; unused missing components removed; CUDA portability changes; no GPU batch16 preflight'}
(OPS/'cpu_acceptance.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n');print(json.dumps(report,ensure_ascii=False))

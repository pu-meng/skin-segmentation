import os,sys,io
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from source.recipe_r1_20261105.recipe import make_optimizer,UpdateSchedule,clip
class Model(torch.nn.Module):
 def __init__(self):
  super().__init__();self.encoder=torch.nn.Linear(2,2);self.decoder=torch.nn.Linear(2,1)
 def forward(self,x):return self.decoder(self.encoder(x))
def step(m,o,s):
 o.zero_grad();loss=m(torch.ones(3,2)).square().mean();loss.backward();clip(m.parameters());s.apply();o.step();s.advance()
torch.manual_seed(42);m=Model();o=make_optimizer(m);s=UpdateSchedule(o);s.update=4364;step(m,o,s)
b=io.BytesIO();torch.save(dict(model=m.state_dict(),optimizer=o.state_dict(),scheduler=s.state_dict(),global_update=s.update),b);b.seek(0);p=torch.load(b,weights_only=False)
n=Model();no=make_optimizer(n);ns=UpdateSchedule(no);n.load_state_dict(p['model'],strict=True);no.load_state_dict(p['optimizer']);ns.load_state_dict(p['scheduler']);assert ns.update==p['global_update']
step(m,o,s);step(n,no,ns)
assert s.update==ns.update
assert all(torch.equal(v,n.state_dict()[k]) for k,v in m.state_dict().items())
assert [g['lr'] for g in o.param_groups]==[g['lr'] for g in no.param_groups]
assert not torch.cuda.is_initialized()
print('PASS real AdamW optimizer+model+global_update save/load yields identical next update across warm/cosine boundary')

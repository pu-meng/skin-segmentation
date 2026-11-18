"""E123.2 fixed-base source x skip matrix; old R1/R3 are read-only references."""
import copy
import torch
from torch import nn
from source.e1232_fourarm_screen_20261106 import candidate as old
ARMS={'BT':'E1232-BT-HAM300-S42','BS':'E1232-BS-HAM300-S42','AU':'E1232-AU-HAM300-S42','BU':'E1232-BU-HAM300-S42'}
class ContentB(old.BottleneckB):
 def __init__(self,base):
  nn.Module.__init__(self)
  self.m=base.m;self.w=base.w;self.context=base.context;self.context_w=copy.deepcopy(base.context)
  self.alpha=nn.Sequential(old.joint.pw(1024,128),nn.GELU(),old.joint.pw(128,512),nn.Sigmoid())
  self.beta=nn.Sequential(old.joint.pw(1024,128),nn.GELU(),old.joint.pw(128,512),nn.Sigmoid())
  for h,ref in [(self.alpha,base.alpha),(self.beta,base.beta)]:
   h[-2].load_state_dict(ref[-2].state_dict());h[0].bias.data.copy_(ref[0].bias.data)
  self.sample=False;self.sampled=None
 def forward(self,x):
  m=self.m(x);w=self.w(x);z=torch.cat([self.context(m),self.context_w(w)],1)
  a=self.alpha(z);b=self.beta(z);y=x+a*m+b*w
  if self.sample:
   with torch.no_grad():
    norm=x.detach().float().norm().clamp_min(1e-12)
    q=torch.tensor([0.,.5,1.],device=x.device)
    self.sampled=torch.cat([torch.quantile(a.detach().float(),q),torch.quantile(b.detach().float(),q),torch.stack([(a*m).detach().float().norm()/norm,(b*w).detach().float().norm()/norm,(y-x).detach().float().norm()/norm])])
  return y
class SkipU(old.joint.E3):
 def forward(self,s,d):
  us=self.s.pre(s);ud=self.d.pre(d);k=self.s.gen(self.s.n1(us));v=self.d.gen(self.d.n1(ud))
  kh=k*(2*self.gs(v.mean((2,3),keepdim=True)).sigmoid())
  return self.s.finish(s,us,kh),d

def build_candidate(arm,pretrained=True):
 assert arm in ARMS
 model=old.build_candidate('R3' if arm=='BS' else 'R1',pretrained)
 if arm[0]=='B':
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(46);model.decoder.e2=ContentB(model.decoder.e2)
 if arm[-1]=='U':
  e=model.decoder.e3;e.__class__=SkipU
  for key in ['mix','n2','ffn','post','gamma']:delattr(e.d,key)
  e.gd=nn.Identity()
 return model

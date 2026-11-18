"""Four-arm E123.2. Original A is reconstructed unchanged before isolated edits."""
import os
from pathlib import Path
import torch
from torch import nn
from source.e123_candidate_a_20261106 import candidate as original
from source.e123_joint_20261105 import candidate as joint
# Imported module requires a valid control identity, not a run launch.
_previous=os.environ.get('BD_ARM');os.environ['BD_ARM']='D4-B3'
try:
 from source.bd_controls_20261105.candidate import BasicMambaResidual
finally:
 if _previous is None:os.environ.pop('BD_ARM',None)
 else:os.environ['BD_ARM']=_previous
from source.three_module_impl_20261104.candidate import DStage
from Models.TinyViM.four_stage_decoder import FourStageDecoder
ARMS={'R0':'A original','R1':'A bottleneck B only','R2':'A Skip S only','R3':'B+S E123.2'}
class BottleneckB(nn.Module):
 def __init__(self):
  super().__init__();self.m=nn.Sequential(*[BasicMambaResidual() for _ in range(3)]);self.w=DStage()
  self.context=nn.Sequential(joint.LN(512),joint.dw(512,3),nn.GELU(),nn.AdaptiveAvgPool2d(1))
  self.alpha=nn.Sequential(joint.pw(512,128),nn.GELU(),joint.pw(128,512),nn.Sigmoid())
  self.beta=nn.Sequential(joint.pw(512,128),nn.GELU(),joint.pw(128,512),nn.Sigmoid())
  for head in [self.alpha,self.beta]:nn.init.zeros_(head[-2].weight);nn.init.constant_(head[-2].bias,-2.197224577336219)
  self.sample=False;self.sampled=None
 def forward(self,x):
  z=self.context(x);a=self.alpha(z);b=self.beta(z);m=self.m(x);w=self.w(x);y=x+a*m+b*w
  if self.sample:
   with torch.no_grad():
    norm=x.detach().float().norm().clamp_min(1e-12)
    self.sampled=torch.cat([torch.quantile(a.detach().float(),torch.tensor([0.,.5,1.],device=x.device)),torch.quantile(b.detach().float(),torch.tensor([0.,.5,1.],device=x.device)),torch.stack([(a*m).detach().float().norm()/norm,(b*w).detach().float().norm()/norm,(y-x).detach().float().norm()/norm])])
  return y
class SkipS(joint.E3):
 def forward(self,s,d):
  us=self.s.pre(s);ud=self.d.pre(d);k=self.s.gen(self.s.n1(us));v=self.d.gen(self.d.n1(ud))
  kh=k*(2*self.gs(v.mean((2,3),keepdim=True)).sigmoid());vh=v*(2*self.gd(v.mean((2,3),keepdim=True)).sigmoid())
  return self.s.finish(s,us,kh),self.d.finish(d,ud,vh)
def build_candidate(arm,pretrained=True):
 assert arm in ARMS
 model=original.build_candidate(pretrained)
 if arm in ['R1','R3']:
  # Regenerate exact baseline D3, not a newly ordered DoubleConv initialization.
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(42);base=FourStageDecoder();model.decoder.stages[3]=base.stages[3]
   torch.manual_seed(45);model.decoder.e2=BottleneckB()
 if arm in ['R2','R3']:model.decoder.e3.__class__=SkipS
 return model

"""Execution v1: Candidate A; replace decoder.stages[2] only. Not a novelty claim."""
import math
import torch
from torch import nn
from source.e123_candidate_a_20261106.candidate import build_candidate as build_a, joint
from mamba_ssm.modules.mamba_simple import Mamba

class WidthMambaResidual(nn.Module):
 """Width-parametrized BasicMambaResidual; unmodified official Mamba-1 internals."""
 def __init__(self,c):
  super().__init__();self.norm=nn.LayerNorm(c);self.mixer=Mamba(d_model=c,d_state=16,d_conv=4,expand=2)
 def correction(self,x):
  b,c,h,w=x.shape;t=x.flatten(2).transpose(1,2)
  return self.mixer(self.norm(t)).transpose(1,2).reshape(b,c,h,w)
 def forward(self,x):return x+self.correction(x)

class DualStage(nn.Module):
 def __init__(self,ci=512,co=256):
  super().__init__();self.project=nn.Conv2d(ci,co,1,bias=True);self.mamba=WidthMambaResidual(co)
  self.rep=nn.Sequential(joint.LN(co),joint.Rep(co,7,False),nn.GELU(),joint.pw(co,co))
  self.raw_m=nn.Parameter(torch.tensor(math.atanh(.1)));self.raw_r=nn.Parameter(torch.tensor(math.atanh(.1)))
 def forward(self,x):
  u=self.project(x)
  # Only pure branch corrections; no repeated addition of the common input.
  return u+.1*self.raw_m.tanh()*self.mamba.correction(u)+.1*self.raw_r.tanh()*self.rep(u)

def build_candidate(pretrained=True):
 model=build_a(pretrained=pretrained)
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(45);model.decoder.stages[2]=DualStage()
 return model

"""Candidate A: replace only joint E2, retain exact original E1 and dual-source E3."""
from pathlib import Path
import torch
from torch import nn
from source.e123_joint_20261105 import candidate as joint
from mamba_ssm.modules.mamba_simple import Mamba
ROOT=Path(__file__).resolve().parents[2]
class OneWayResidual(nn.Module):
 def __init__(self):
  super().__init__();self.norm=nn.LayerNorm(128);self.mixer=Mamba(d_model=128,d_state=16,d_conv=4,expand=2,dt_rank='auto')
 def forward(self,x):
  b,c,h,w=x.shape;t=x.flatten(2).transpose(1,2);t=t+self.mixer(self.norm(t));return t.transpose(1,2).reshape(b,c,h,w)
class E2A(nn.Module):
 def __init__(self):
  super().__init__();self.pre=joint.surround(512,256)
  self.spatial=nn.Sequential(*[OneWayResidual() for _ in range(3)])
  self.channel=nn.Sequential(nn.Linear(3,16),nn.GELU(),nn.Linear(16,1))
  self.select=nn.Sequential(nn.Linear(256,32),nn.GELU(),nn.Linear(32,256))
  for m in [self.channel[-1],self.select[-1]]:nn.init.zeros_(m.weight);nn.init.zeros_(m.bias)
  self.fuse=joint.pw(256,256);self.norm=joint.LN(256)
  self.ffn=nn.Sequential(joint.pw(256,512),nn.GELU(),joint.pw(512,256))
  self.activation=nn.GELU();self.output=joint.pw(256,512);self.gamma=nn.Parameter(torch.full((1,512,1,1),.01))
 @staticmethod
 def spectral_statistics(x):
  with torch.autocast(x.device.type,enabled=False):
   f=torch.fft.fft2(x.float(),norm='ortho');power=f.real.square()+f.imag.square();h,w=x.shape[-2:]
   fy=torch.fft.fftfreq(h,device=x.device);fx=torch.fft.fftfreq(w,device=x.device)
   mask=(fy[:,None].abs()<=.25)&(fx[None,:].abs()<=.25)
   el=(power*mask).sum((-2,-1));eh=(power*(~mask)).sum((-2,-1));rh=eh/(el+eh+1e-8)
   return torch.stack([torch.log1p(el),torch.log1p(eh),rh],-1)
 def forward(self,x):
  us,uc=self.pre(x).chunk(2,1);fs=self.spatial(us)
  a=2*self.channel(self.spectral_statistics(uc)).squeeze(-1).sigmoid();fc=uc*a[:,:,None,None]
  z=torch.cat([fs.mean((2,3)),fc.mean((2,3))],1)
  weights=self.select(z).reshape(-1,2,128,1,1).softmax(1)
  fused=self.fuse(torch.cat([2*weights[:,0]*fs,2*weights[:,1]*fc],1))
  return x+self.gamma*self.output(self.activation(self.ffn(self.norm(fused))))
def build_candidate(pretrained=True):
 model=joint.build_candidate(pretrained=pretrained)
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(44);model.decoder.e2=E2A()
 return model

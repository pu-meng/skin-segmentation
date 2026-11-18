"""Isolated first-batch E2 candidates; original DualA and shared sources untouched."""
import torch
from torch import nn
from source.decoder_dual_a_20261106.candidate import build_candidate as base,WidthMambaResidual,joint
from official_afno2d import AFNO2D
NAMES={'NoE2':'DualA｜瓶颈恒等','Input':'DualA｜局部全局·输入选路','Cross':'DualA｜局部查询全局','AFNO':'DualA｜局部傅里叶混合'}
class FourierCorrection(nn.Module):
 def __init__(self):
  super().__init__();self.op=AFNO2D(128,num_blocks=8,sparsity_threshold=.01,hard_thresholding_fraction=1,hidden_size_factor=1)
 def forward(self,x):
  # [B,128,H,W] -> [B,H*W,128] -> official residual output minus input -> pure correction.
  b,c,h,w=x.shape;t=x.flatten(2).transpose(1,2);y=self.op(t,spatial_size=(h,w))-t
  return y.transpose(1,2).reshape(b,c,h,w)
class E2(nn.Module):
 def __init__(self,kind):
  super().__init__();self.kind=kind
  # Shared construction order and seed preserve local branch/output initialization across candidates.
  self.pre=nn.Sequential(joint.LN(512),joint.pw(512,256))
  self.local=nn.Sequential(joint.LN(128),joint.dw(128,7),nn.GELU(),joint.pw(128,128))
  self.global_norm=joint.LN(128);self.global_op=WidthMambaResidual(128)
  self.select=nn.Sequential(joint.pw(256,32),nn.GELU(),joint.pw(32,256))
  nn.init.zeros_(self.select[-1].weight);nn.init.zeros_(self.select[-1].bias)
  self.output=nn.Sequential(joint.LN(256),nn.GELU(),joint.pw(256,512));self.gamma=nn.Parameter(torch.full((1,512,1,1),.01))
  if kind=='Cross':
   del self.select
   with torch.random.fork_rng(devices=[]):
    torch.manual_seed(49);self.q=joint.pw(128,128);self.k=joint.pw(128,128);self.v=joint.pw(128,128)
  elif kind=='AFNO':
   with torch.random.fork_rng(devices=[]):torch.manual_seed(50);self.global_op=FourierCorrection()
 def contents(self,z):
  a,b=z.chunk(2,1);l=self.local(a);b=self.global_norm(b)
  g=self.global_op(b) if self.kind=='AFNO' else self.global_op.correction(b)
  return l,g
 def fuse(self,z,l,g):
  if self.kind=='Cross':
   # [B,128,8,8] -> [B,4,64,32]; attention matches spatial positions, not channels.
   b,c,h,w=l.shape;n=h*w
   q=self.q(l).reshape(b,4,32,n).transpose(-2,-1)
   k=self.k(g).reshape(b,4,32,n).transpose(-2,-1)
   v=self.v(g).reshape(b,4,32,n).transpose(-2,-1)
   weights=(q@k.transpose(-2,-1)/(32**.5)).softmax(-1)
   read=(weights@v).transpose(-2,-1).reshape(b,128,h,w)
   return torch.cat([l,read],1)
  weights=self.select(z.mean((2,3),keepdim=True)).reshape(-1,2,128,1,1).softmax(1)
  return torch.cat([2*weights[:,0]*l,2*weights[:,1]*g],1)
 def forward(self,x):
  z=self.pre(x);l,g=self.contents(z);return x+self.gamma*self.output(self.fuse(z,l,g))
def build(arm,pretrained=True):
 assert arm in NAMES
 model=base(pretrained)
 if arm=='NoE2':model.decoder.e2=nn.Identity()
 else:
  with torch.random.fork_rng(devices=[]):torch.manual_seed(48);model.decoder.e2=E2(arm)
 return model

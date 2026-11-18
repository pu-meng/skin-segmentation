"""Original E3.1 math at one selected skip; C varies with the aligned level."""
import torch
from torch import nn
from torch.nn import functional as F
from source.three_module_e23_20261104.candidate import FullScan,Gate
from Models.E2Simplified.model import RepDW7
from Models.E3DynamicSkip.vendor.fdconv import FDConv

class DynamicSkip(nn.Module):
 def __init__(self,channels=64):
  super().__init__();self.channels=channels
  self.global_path=FullScan(channels)
  self.repdw=RepDW7(channels)
  # Official FDW+global/local KSM+internal FBM all enabled. No external band split.
  self.fdconv=FDConv(channels,channels,3,stride=1,padding=1,groups=1,bias=True,kernel_num=4,
    use_fdconv_if_c_gt=16,use_fdconv_if_k_in=[1,3],use_fdconv_if_stride_in=[1],
    use_fbm_if_k_in=[3],use_fbm_for_stride=False,use_ksm_local=True,
    convert_param=True,param_ratio=1,param_reduction=1.0,
    ksm_global_act='sigmoid',ksm_local_act='sigmoid',kernel_temp=1.0,
    spatial_freq_decompose=False)
  self.local_fuse=nn.Conv2d(2*channels,channels,1)
  self.g_global=Gate(channels,channels);self.g_local=Gate(channels,channels)
  self.select=nn.Conv2d(channels,2,1);nn.init.zeros_(self.select.weight);nn.init.zeros_(self.select.bias)
  self.decoder_project=nn.Conv2d(channels,channels,1)
  self.eta=nn.Parameter(torch.full((1,channels,1,1),.001))
 def forward(self,s,d,enabled=True):
  if s.shape!=d.shape or s.shape[1]!=self.channels:raise ValueError('S and D must share B,C,H,W interface')
  if not enabled:return s
  a=self.global_path(s);b=self.local_fuse(torch.cat([self.repdw(s),self.fdconv(s)],1))
  ag=self.g_global(a,b);bg=self.g_local(b,a)
  weights=self.select(d).softmax(1)
  selected=weights[:,0:1]*ag+weights[:,1:2]*bg
  return s+self.eta*(selected+self.decoder_project(d))

class Decoder(nn.Module):
 def __init__(self,base,level):
  super().__init__();self.stages=base.stages;self.ups=base.ups;self.head=base.head
  self.level=level;self.refiner=DynamicSkip(64*2**level);self.enabled=True
 def forward(self,features):
  x=self.stages[3](features[3])
  for level in (2,1,0):
   x=self.ups[level](x);s=features[level]
   if level==self.level:s=self.refiner(s,x,enabled=self.enabled)
   x=self.stages[level](torch.cat([x,s],1))
  return torch.sigmoid(F.interpolate(self.head(x),size=(256,256),mode='bilinear',align_corners=False))

def build_candidate(pretrained=True,level=0,enabled=True):
 from Models.TinyViM.four_stage_decoder import build_four_stage_baseline
 m=build_four_stage_baseline(pretrained=pretrained)
 if not enabled:return m
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(42);m.decoder=Decoder(m.decoder,level)
 return m

"""B-WPO-MR: parallel full-channel E1 and split Mamba/RepDW, static channel scaling."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
import torch
from torch import nn
from torch.nn import functional as F
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t');prepare_imports('tinyvim_s')
from mamba_ssm.modules.mamba_simple import Mamba
from Models.TinyViM.blocks import RepDW
from source.wpo_mr_20261105.e1_wave import DStage
class SplitMambaRepDW(nn.Module):
 def __init__(self):
  super().__init__();self.norm=nn.LayerNorm(256);self.mixer=Mamba(d_model=256,d_state=16,d_conv=4,expand=2);self.local=RepDW(256)
 def forward(self,x):
  a,b=x.split(256,dim=1);n,c,h,w=a.shape;t=a.flatten(2).transpose(1,2);a=(t+self.mixer(self.norm(t))).transpose(1,2).reshape(n,c,h,w)
  return torch.cat([a,self.local(b)],dim=1)
class ParallelBottleneck(nn.Module):
 def __init__(self):
  super().__init__();self.a=DStage();self.b=SplitMambaRepDW();self.c=nn.Parameter(torch.ones(1,1024,1,1));self.projection=nn.Conv2d(1024,512,1)
  self.post=nn.Sequential(nn.Conv2d(512,512,3,padding=1,groups=512,bias=False),nn.Conv2d(512,512,1,bias=False),nn.BatchNorm2d(512),nn.GELU())
 def forward(self,x):
  assert tuple(x.shape[1:])==(512,8,8)
  a=self.a(x);b=self.b(x);h=x+self.projection(self.c*torch.cat([a,b],dim=1));return self.post(h)
class Decoder(nn.Module):
 def __init__(self,old):
  super().__init__();self.stages=old.stages;self.ups=old.ups;self.head=old.head;self.enhance=ParallelBottleneck()
 def forward(self,f):
  assert [tuple(v.shape[1:]) for v in f]==[(64,64,64),(128,32,32),(256,16,16),(512,8,8)]
  x=self.stages[3](self.enhance(f[3]))
  for level in (2,1,0):x=self.stages[level](torch.cat([self.ups[level](x),f[level]],1))
  return torch.sigmoid(F.interpolate(self.head(x),size=(256,256),mode='bilinear',align_corners=False))
def build_candidate(pretrained=True,enabled=True):
 from Models.TinyViM.four_stage_decoder import build_four_stage_baseline
 model=build_four_stage_baseline(pretrained=pretrained)
 if enabled:
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(42);model.decoder=Decoder(model.decoder)
 return model
def added_parameters(model):
 for n,p in model.named_parameters():
  if n.startswith('decoder.enhance.'):yield n,p

from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
import torch
from torch import nn
from torch.nn import functional as F
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t');prepare_imports('tinyvim_s')
from source.b3_e1_e31_20261105.components import BasicMambaResidual,DStage
from Models.E3DynamicSkip.model import DynamicSkip
class Decoder(nn.Module):
 def __init__(self,old):
  super().__init__();self.stages=old.stages;self.ups=old.ups;self.head=old.head
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(44);self.enhance=nn.Sequential(*[BasicMambaResidual() for _ in range(3)])
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(42);self.stages[3]=DStage()
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(42);self.refiner=DynamicSkip()
 def forward(self,f):
  assert [tuple(v.shape[1:]) for v in f]==[(64,64,64),(128,32,32),(256,16,16),(512,8,8)]
  x=self.stages[3](self.enhance(f[3]))
  for level in (2,1,0):
   d=self.ups[level](x);s=f[level]
   if level==0:s=self.refiner(s,d)
   x=self.stages[level](torch.cat([d,s],1))
  return torch.sigmoid(F.interpolate(self.head(x),size=(256,256),mode='bilinear',align_corners=False))
def build_candidate(pretrained=True,enabled=True):
 from Models.TinyViM.four_stage_decoder import build_four_stage_baseline
 model=build_four_stage_baseline(pretrained=pretrained)
 if enabled:model.decoder=Decoder(model.decoder)
 return model
def added_parameters(model):
 for n,p in model.named_parameters():
  if n.startswith(('decoder.enhance.','decoder.stages.3.','decoder.refiner.')):yield n,p

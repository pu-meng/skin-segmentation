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
from source.b3d_repeat_20261106.selection import COUNT,MED
from source.bd_controls_20261105.official_mednext import MedNeXtBlock
class BasicMambaResidual(nn.Module):
 def __init__(self):
  super().__init__();self.norm=nn.LayerNorm(512);self.mixer=Mamba(d_model=512,d_state=16,d_conv=4,expand=2)
 def forward(self,x):
  b,c,h,w=x.shape;t=x.flatten(2).transpose(1,2);t=t+self.mixer(self.norm(t));return t.transpose(1,2).reshape(b,c,h,w)
class MedStage(nn.Sequential):
 def __init__(self,ci,co):
  super().__init__(nn.Conv2d(ci,co,1) if ci!=co else nn.Identity(),*[MedNeXtBlock(co,co,exp_r=4,kernel_size=7,do_res=True,norm_type='group',dim='2d',grn=False) for _ in range(2)])
class Decoder(nn.Module):
 def __init__(self,old,count,med):
  super().__init__();self.stages=old.stages;self.ups=old.ups;self.head=old.head;self.enhance=nn.Sequential(*[BasicMambaResidual() for _ in range(count)])
  if med:self.stages=nn.ModuleList([MedStage(ci,co) for ci,co in [(128,64),(256,128),(512,256),(512,512)]])
 def forward(self,f):
  assert [tuple(v.shape[1:]) for v in f]==[(64,64,64),(128,32,32),(256,16,16),(512,8,8)]
  x=self.stages[3](self.enhance(f[3]))
  for level in (2,1,0):x=self.stages[level](torch.cat([self.ups[level](x),f[level]],1))
  return torch.sigmoid(F.interpolate(self.head(x),size=(256,256),mode='bilinear',align_corners=False))
def build_candidate(pretrained=True,count=None,med=None):
 from Models.TinyViM.four_stage_decoder import build_four_stage_baseline
 count=COUNT if count is None else count;med=MED if med is None else med
 model=build_four_stage_baseline(pretrained=pretrained)
 if count or med:
  with torch.random.fork_rng(devices=[]):torch.manual_seed(44);model.decoder=Decoder(model.decoder,count,med)
 return model
def added_parameters(model):
 for n,p in model.named_parameters():
  if n.startswith('decoder.enhance.') or (MED and n.startswith('decoder.stages.')):yield n,p

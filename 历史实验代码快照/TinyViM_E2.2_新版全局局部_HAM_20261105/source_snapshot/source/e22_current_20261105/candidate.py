from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
import torch
from torch import nn
from torch.nn import functional as F
from source.e123_joint_20261105.candidate import E2
class Decoder(nn.Module):
 def __init__(self,old):
  super().__init__();self.stages=old.stages;self.ups=old.ups;self.head=old.head;self.enhance=E2()
 def forward(self,f):
  x=self.stages[3](self.enhance(f[3]))
  for level in (2,1,0):x=self.stages[level](torch.cat([self.ups[level](x),f[level]],1))
  return torch.sigmoid(F.interpolate(self.head(x),size=(256,256),mode='bilinear',align_corners=False))
def build_candidate(pretrained=True):
 from Models.TinyViM.four_stage_decoder import build_four_stage_baseline
 model=build_four_stage_baseline(pretrained=pretrained)
 with torch.random.fork_rng(devices=[]):torch.manual_seed(42);model.decoder=Decoder(model.decoder)
 return model

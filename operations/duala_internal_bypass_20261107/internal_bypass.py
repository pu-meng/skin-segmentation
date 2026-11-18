"""Prepared alternative: internal per-channel bypasses, no live training patch."""
import sys
from pathlib import Path
import torch
from torch import nn
OLD=Path('/home/PuMengYu/experiments/skin_segmentation/operations/duala_E2_batch1_20261107')
sys.path.insert(0,str(OLD))
from candidates import E2,build as old_build
class InternalBypassE2(E2):
 def add_bypasses(self):
  # Static learned channel scales, not input-conditioned gates. Nonzero init permits immediate branch gradients.
  self.depth_scale=nn.Parameter(torch.full((1,128,1,1),.1))
  self.point_scale=nn.Parameter(torch.full((1,128,1,1),.1))
  self.global_scale=nn.Parameter(torch.full((1,128,1,1),.1))
  return self
 def local_contents(self,a):
  # a:[B,128,H,W]; A=LN+DW7; bypass A before B=GELU+PW.
  h=a+self.depth_scale*self.local[1](self.local[0](a))
  l=h+self.point_scale*self.local[3](self.local[2](h))
  return l
 def global_contents(self,b):
  # Keep raw branch input b, rather than normalized b, on the identity path.
  return b+self.global_scale*self.global_op.correction(self.global_norm(b))
 def contents(self,z):
  a,b=z.chunk(2,1);return self.local_contents(a),self.global_contents(b)
def build(pretrained=True):
 model=old_build('Input',pretrained);e=model.decoder.e2;e.__class__=InternalBypassE2;e.add_bypasses();return model

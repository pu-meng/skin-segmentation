"""E3.2: original complete E3; Decoder-only soft box weights its correction."""
from pathlib import Path
import sys, importlib.util
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
spec=importlib.util.spec_from_file_location('_original_e3',ROOT/'source/three_module_e23_20261104/candidate.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
MIN_SIZE=1/256
TAU=2/64

def legal_box(raw):
 wh=MIN_SIZE+(1-MIN_SIZE)*torch.sigmoid(raw[:,2:])
 center=wh/2+(1-wh)*torch.sigmoid(raw[:,:2])
 return torch.cat([center,wh],1)

def soft_rectangle(box,h,w):
 # Sample feature pixel centres in normalized image coordinates. FP32 edges.
 box=box.float();cx,cy,bw,bh=box.unbind(1)
 xx=(torch.arange(w,device=box.device,dtype=torch.float32)+.5)/w
 yy=(torch.arange(h,device=box.device,dtype=torch.float32)+.5)/h
 left,right=cx-bw/2,cx+bw/2;top,bottom=cy-bh/2,cy+bh/2
 wx=torch.sigmoid((xx[None,:]-left[:,None])/TAU)*torch.sigmoid((right[:,None]-xx[None,:])/TAU)
 wy=torch.sigmoid((yy[None,:]-top[:,None])/TAU)*torch.sigmoid((bottom[:,None]-yy[None,:])/TAU)
 return wy[:,None,:,None]*wx[:,None,None,:]

def mask_boxes(mask):
 # Called on current already-augmented GT only. Inclusive pixel extent.
 b,_,h,w=mask.shape;out=mask.new_zeros((b,4));valid=torch.zeros(b,device=mask.device,dtype=torch.bool)
 for i in range(b):
  y,x=torch.where(mask[i,0]>.5)
  if len(x):
   l=x.min().float()/w;r=(x.max().float()+1)/w;t=y.min().float()/h;d=(y.max().float()+1)/h
   out[i]=torch.stack([(l+r)/2,(t+d)/2,r-l,d-t]);valid[i]=True
  else:out[i]=mask.new_tensor([.5,.5,1.,1.])
 return out,valid

def auxiliary_box_loss(box,mask):
 target,valid=mask_boxes(mask)
 # Empty GT: omit box regression, segmentation remains supervised.
 if valid.any():return F.smooth_l1_loss(box[valid],target[valid],beta=.1)
 return box.sum()*0

class BoxDecoder(nn.Module):
 def __init__(self,decoder):
  super().__init__();self.stages=decoder.stages;self.ups=decoder.ups;self.head=decoder.head;self.enhance=decoder.enhance
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(44)
   self.box_head=nn.Sequential(nn.Conv2d(64,32,3,padding=1),nn.GELU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(32,4))
  self.constraint_enabled=True;self.last_box=None;self.last_weight=None
 def forward(self,f):
  x=self.stages[3](f[3])
  for level in (2,1):x=self.stages[level](torch.cat([self.ups[level](x),f[level]],1))
  d=self.ups[0](x) # Already computed D1 upsample/alignment, never D0 output.
  box=legal_box(self.box_head(d));w=soft_rectangle(box,*f[0].shape[-2:])
  self.last_box=box;self.last_weight=w
  e=self.enhance(f[0]) # All original E3 operations unchanged.
  # Direct e at W=1 avoids subtraction/readdition rounding; mathematically identical.
  s=f[0]+w*(e-f[0]) if self.constraint_enabled else e
  x=self.stages[0](torch.cat([d,s],1))
  return torch.sigmoid(F.interpolate(self.head(x),size=(256,256),mode='bilinear',align_corners=False))

def build_candidate(pretrained=True):
 model=old.build_candidate(pretrained=pretrained,arm='E3')
 model.decoder=BoxDecoder(model.decoder)
 return model

class TrainingCriterion(nn.Module):
 def __init__(self,model,segmentation):
  super().__init__();object.__setattr__(self,'network',model);self.segmentation=segmentation
 def forward(self,prediction,target):
  loss=self.segmentation(prediction,target)
  if self.network.training:loss=loss+.1*auxiliary_box_loss(self.network.decoder.last_box,target)
  return loss

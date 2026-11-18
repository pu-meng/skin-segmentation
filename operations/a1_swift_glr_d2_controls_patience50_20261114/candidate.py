import torch
from Models.A1SwiftFormerGLR.model import D2GLR as Anchor
from Models.A1SwiftFormerGLRv12.model import D2GLR as C3
from Models.A1SwiftFormerGLRControls.model import D2GLR
from operations.a1swift_pair_ablation_20261112.entry import build as build_baseline

def build_candidate(variant,pretrained=True):
 model=build_baseline('no_e2_no_d2d3',pretrained=pretrained)
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(20261114);anchor=Anchor('C1').state_dict()
  torch.manual_seed(20261114);c3=C3('C3');state=c3.state_dict()
  for k in state:
   if not k.startswith('block.refine.'):state[k]=anchor[k].clone()
  torch.manual_seed(20261114);new=D2GLR(variant);dst=new.state_dict()
  for k in dst:
   if k.startswith(('block.select_dw.','block.select_norm.','block.select_out.')):continue
   assert k in state and state[k].shape==dst[k].shape,k
   dst[k]=state[k].clone()
  new.load_state_dict(dst,strict=True);model.decoder.stages[2]=new
 return model

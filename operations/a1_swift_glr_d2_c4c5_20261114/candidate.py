"""C4/C5 preserve C3 non-branch parameters and the applicable branch weights."""
import torch
from Models.A1SwiftFormerGLR.model import D2GLR as AnchorD2
from Models.A1SwiftFormerGLRv12.model import D2GLR as C3D2
from Models.A1SwiftFormerGLRv12C45.model import D2GLR
from operations.a1swift_pair_ablation_20261112.entry import build as build_baseline

def build_candidate(variant,pretrained=True):
 model=build_baseline('no_e2_no_d2d3',pretrained=pretrained)
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(20261114); anchor=AnchorD2('C1')
  torch.manual_seed(20261114); c3=C3D2('C3')
  state=c3.state_dict();old=anchor.state_dict()
  for k in state:
   if not k.startswith('block.refine.'):state[k]=old[k].clone()
  c3.load_state_dict(state,strict=True)
  torch.manual_seed(20261114);new=D2GLR(variant);dst=new.state_dict()
  for k in dst:
   source=k
   if variant=='C4' and k.startswith('block.branches.1.'):source=k.replace('block.branches.1.','block.branches.0.',1)
   elif variant=='C4' and k.startswith('block.branches.2.'):source=k.replace('block.branches.2.','block.branches.1.',1)
   assert source in state and state[source].shape==dst[k].shape, (k,source)
   dst[k]=state[source].clone()
  new.load_state_dict(dst,strict=True);model.decoder.stages[2]=new
 return model

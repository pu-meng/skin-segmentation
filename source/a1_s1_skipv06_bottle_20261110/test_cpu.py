import sys, copy, dataclasses
sys.path[:0]=['/home/PuMengYu/experiments/skin_segmentation','/home/PuMengYu/experiments/skin_segmentation/source/comparison','/home/PuMengYu/experiments/skin_segmentation/source/bottleneck_mamba']
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t'); prepare_imports('tinyvim_s')
import torch
from source.a1_s1_skipv06_bottle_20261110.candidate import BottleneckBottle, BottleConfig, DirectionalWPO, build_candidate
from source.a1_s1_skip1_v06_20261108.candidate import build_candidate as parent

torch.set_num_threads(2); torch.manual_seed(42); device='cpu'
for flag in ['all','enable_mamba','enable_global_query','enable_dwpo','enable_input_repdw','enable_output_repdw','enable_stage4']:
 cfg=BottleConfig() if flag=='all' else dataclasses.replace(BottleConfig(),**{flag:False})
 m=BottleneckBottle(cfg).to(device).train(); m.mamba=torch.nn.Linear(64,64);  x=torch.randn(2,512,8,8,device=device,requires_grad=True)
 def forbidden(*args): raise AssertionError('disabled branch was called')
 handles=[]
 for name,on in [('mamba',cfg.enable_mamba),('query_proj',cfg.enable_global_query),('query_mlp',cfg.enable_global_query),('wave',cfg.enable_dwpo),('stage4_fuse',cfg.enable_stage4)]:
  if not on: handles.append(getattr(m,name).register_forward_pre_hook(forbidden))
 y=m(x); y.square().mean().backward()
 assert y.shape==x.shape and torch.isfinite(y).all() and torch.isfinite(x.grad).all()
 assert m.mamba_conv[0].spatial.weight.grad is not None and m.wave_conv[0].spatial.weight.grad is not None
 assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
 print(flag,'forward_backward_pass')
 del m,x,y
m=BottleneckBottle(BottleConfig(enable_mamba=False)).to(device).eval()
with torch.no_grad():
 m.gamma_w.zero_()
 captured=[]
 handle=m.wave_expand.register_forward_hook(lambda module, inputs, output: captured.append(output.detach()))
 m(torch.randn(2,512,8,8,device=device)); handle.remove()
 assert len(captured)==1 and torch.count_nonzero(captured[0])==0
print('zero gamma_w produces exactly zero projected update')
w=DirectionalWPO().to(device); x=torch.randn(2,128,8,8,device=device); w.directional=False; a=w(x); w.directional=True; b=w(x)
assert torch.equal(a,b); print('isotropic_equal')
m=BottleneckBottle(BottleConfig(enable_mamba=False)).to(device).eval(); x=torch.randn(2,512,8,8,device=device)
with torch.no_grad(): a=m(x); m.reparameterize_repdw(); b=m(x)
print('fusion max error', (a-b).abs().max().item()); torch.testing.assert_close(a,b,rtol=1e-5,atol=2e-6)
torch.manual_seed(42); base=parent(pretrained=False)
torch.manual_seed(42); model=build_candidate(pretrained=False)
a=base.state_dict(); b=model.state_dict(); same=[k for k in a if not k.startswith('decoder.e2.')]; assert all(k in b and torch.equal(a[k],b[k]) for k in same); assert all(k.startswith('decoder.e2.') or k in a for k in b); print('parent preservation',len(same))
print('actual Mamba and full network forward require CUDA and remain unverified')

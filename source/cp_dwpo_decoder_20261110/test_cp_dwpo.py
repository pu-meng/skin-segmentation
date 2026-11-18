import sys
sys.path[:0]=['/home/PuMengYu/experiments/skin_segmentation','/home/PuMengYu/experiments/skin_segmentation/source/comparison','/home/PuMengYu/experiments/skin_segmentation/source/bottleneck_mamba']
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t');prepare_imports('tinyvim_s')
import torch
from source.cp_dwpo_decoder_20261110.candidate import dct2,idct2,Wave,build_candidate
torch.set_num_threads(2)
for shape in [(2,3,7,11),(1,4,8,8),(2,5,16,32)]:
 x=torch.randn(*shape,requires_grad=True); y=idct2(dct2(x)); error=(y-x).abs().max().item(); assert error<2e-5; y.square().mean().backward(); assert torch.isfinite(x.grad).all(); print('DCT',shape,error,flush=True)
w=Wave(8).cuda(); a=torch.randn(2,8,7,11,device='cuda',requires_grad=True); b=w(a);b.square().mean().backward();assert torch.isfinite(a.grad).all() and all(p.grad is not None and torch.isfinite(p.grad).all() for p in w.parameters());print('dynamic wave gradients passed',flush=True)
for base in ('skip','bottle'):
 torch.manual_seed(42);m=build_candidate(False,base).cuda().train();x=torch.randn(2,3,256,256,device='cuda');torch.cuda.reset_peak_memory_stats()
 for mixed in (False,True):
  m.zero_grad(set_to_none=True)
  with torch.autocast('cuda',enabled=mixed): y=m(x); loss=y.float().square().mean()
  loss.backward(); assert y.shape==(2,1,256,256) and torch.isfinite(y).all() and all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
  torch.cuda.synchronize();print(base,'amp',mixed,'peak_GiB',round(torch.cuda.max_memory_allocated()/2**30,3),flush=True)
 del m,x,y,loss;torch.cuda.empty_cache()

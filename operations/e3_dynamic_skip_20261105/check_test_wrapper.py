import os,sys,json
from pathlib import Path
R=Path('/home/PuMengYu/experiments/skin_segmentation');sys.path[:0]=[str(R/'source/e3_dynamic_skip')]
import bootstrap
import torch
from experiment import OUT,build_candidate,write
sys.path.insert(0,str(R/'operations/nnunet_mirror_validation'))
from run_validation import Predictor
torch.set_num_threads(4);torch.manual_seed(42)
m=build_candidate(False).cuda().eval();x=torch.rand(1,3,256,256,device='cuda')
with torch.no_grad():
 plain=m(x);p=Predictor(m,False).eval();assert torch.equal(plain,p(x));p.close()
 p=Predictor(m,True).eval();y=p(x);assert p.verify_remaining==0 and p.max_error==0 and torch.isfinite(y).all();p.close()
 s=torch.randn(2,64,8,8,device='cuda');d=torch.randn_like(s);f=m.decoder.refiner
 a=f.global_path(s);b=f.local_fuse(torch.cat([f.repdw(s),f.fdconv(s)],1));ag=f.g_global(a,b);bg=f.g_local(b,a);w=f.select(d).softmax(1)
 ref=s+f.eta*(w[:,0:1]*ag+w[:,1:2]*bg+f.decoder_project(d))
 assert torch.equal(ref,f(s,d))
write(OUT/'test_wrapper_preflight.json',dict(passed=True,real_head_logits_exact=True,none_bitwise_equals_original=True,mirror4_finite=True,actual_views=[[],[2],[3],[2,3]],manual_residual_formula_exact=True,decoder_D_projected_and_selected=True))
print('TEST_WRAPPER_PREFLIGHT_PASS',flush=True)

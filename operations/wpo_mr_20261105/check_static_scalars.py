import os,sys,json
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from source.wpo_mr_20261105.candidate import ParallelBottleneck
from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
import mamba_ssm.modules.mamba_simple as ms
ms.causal_conv1d_fn=None;ms.selective_scan_fn=selective_scan_ref
torch.set_num_threads(4);torch.manual_seed(42);m=ParallelBottleneck();x=torch.randn(2,512,8,8,requires_grad=True);y=m(x);loss=(y*torch.randn_like(y)).mean();loss.backward()
g=m.c.grad;assert g.numel()==1024 and torch.isfinite(g).all() and (g!=0).all()
old=m.c.detach().clone();lr=1e-4/float(g.abs().min());torch.optim.SGD([m.c],lr=lr,weight_decay=0).step();assert (m.c.detach()!=old).all()
a=dict(passed=True,device='cpu',batch=2,c_numel=1024,c_gradient_nonzero_scalars=int((g!=0).sum()),diagnostic_only_SGD_lr=lr,lr_rule='min scalar update1e-4, not formal training lr',all_c_updated_SGD_without_weight_decay=True,no_cuda_initialized=not torch.cuda.is_initialized())
assert a['no_cuda_initialized'];Path(__file__).with_suffix('.json').write_text(json.dumps(a,indent=2));print(json.dumps(a))

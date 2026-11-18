import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
p=Path(__file__).parent/'candidate_model.py'
assert p.exists(), 'Figure2b candidate full network is not implemented'
from candidate_model import build_model, SS2D
import torch
m=build_model(cpu_reference=True)
assert sum(isinstance(x,SS2D) for x in m.modules())==56
assert len([p for n,p in m.named_parameters() if n.endswith('skip_scale') and float(p)==0])==6
x=torch.randn(2,3,64,64,requires_grad=True)
y=m(x)
assert y.shape==(2,1,64,64) and torch.isfinite(y).all()
y.mean().backward()
assert x.grad is not None and torch.isfinite(x.grad).all()
assert not torch.cuda.is_initialized()
print('CPU full network behavior passed')

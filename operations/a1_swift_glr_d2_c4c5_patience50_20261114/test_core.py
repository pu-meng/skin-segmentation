import torch
from torch import nn
from Models.A1SwiftFormerGLRv12C45.model import GLR

torch.set_num_threads(1)
for name in ('C4','C5'):
 m=GLR(16,name);x=torch.randn(2,16,8,8,requires_grad=True);y,a=m.forward_with_weights(x)
 assert y.shape==x.shape and a.shape==(2,4,8,8) and torch.allclose(a,torch.full_like(a,.25))
 if name=='C4':assert isinstance(m.branches[0],nn.Identity) and m.branches[1][0].kernel_size==(3,3) and m.branches[2][0].kernel_size==(7,7)
 if name=='C5':assert all(not any(isinstance(q,nn.GELU) for q in b.modules()) for b in m.branches) and isinstance(m.pre[-1],nn.GELU)
 y.square().mean().backward();assert torch.isfinite(x.grad).all() and all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
 with torch.no_grad():m.select_out.weight.normal_(0,.01)
 m.zero_grad();m(x.detach()).square().mean().backward();assert all(any(p.grad.abs().sum()>0 for p in b.parameters()) for b in m.descriptors)
 print(name,'CPU_ACCEPTANCE_PASS')

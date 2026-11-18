import torch
from Models.A1SwiftFormerGLRControls.model import GLR
torch.set_num_threads(1)
for variant in ('CTRL-U','CTRL-AVG'):
 m=GLR(16,variant);x=torch.randn(2,16,8,8,requires_grad=True);y,a=m.forward_with_weights(x)
 assert y.shape==x.shape and a.shape==(2,4,8,8) and torch.allclose(a,torch.full_like(a,.25))
 assert not hasattr(m,'descriptors')
 if variant=='CTRL-AVG':assert not hasattr(m,'select_out') and not hasattr(m,'context_project')
 y.square().mean().backward(); assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
 if variant=='CTRL-U':
  with torch.no_grad():m.select_out.weight.normal_(0,.01)
  m.zero_grad();m(x.detach()).square().mean().backward();assert m.context_project.weight.grad.abs().sum()>0
 print(variant,'CPU_ACCEPTANCE_PASS')

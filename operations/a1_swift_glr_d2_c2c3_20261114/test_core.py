import torch
from Models.A1SwiftFormerGLRv12.model import GLR,SplitFFN
from Models.A1SwiftFormerGLR.model import D2GLR as Anchor
from operations.a1_swift_glr_d2_c2c3_20261114.candidate import D2GLR

torch.set_num_threads(1)
for variant in ('C2','C3'):
    m=GLR(16,variant); x=torch.randn(2,16,8,8,requires_grad=True)
    y,a=m.forward_with_weights(x)
    assert y.shape==x.shape and a.shape==(2,4,8,8) and torch.allclose(a,torch.full_like(a,.25))
    assert m.gamma_c.item()>0 and m.gamma_s.item()>0
    y.square().mean().backward()
    assert torch.isfinite(x.grad).all() and all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    assert all(any(p.grad.abs().sum()>0 for p in b.parameters()) for b in m.branches)
    with torch.no_grad(): m.select_out.weight.normal_(0, 0.01)
    m.zero_grad(); m(x.detach()).square().mean().backward()
    assert all(any(p.grad.abs().sum()>0 for p in b.parameters()) for b in m.descriptors)
    print(variant,'shape null equal weights branches descriptors gradient finite passed')
try: SplitFFN(15)
except ValueError: pass
else: raise AssertionError('odd channels should be rejected')
print('C2_C3_CPU_ACCEPTANCE_PASS; odd channels rejected')

"""CPU acceptance for the first two GLR variants only."""
import sys
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation');sys.path.insert(0,str(ROOT))
import torch
from Models.A1SwiftFormerGLR.model import GLR,D2GLR

def main():
    torch.set_num_threads(2)
    torch.manual_seed(123);a=D2GLR('C0')
    torch.manual_seed(123);b=D2GLR('C1')
    sd0=a.state_dict();sd1=b.state_dict()
    shared=[k for k in sd0 if not k.startswith('block.select_out.')]
    assert all(torch.equal(sd0[k],sd1[k]) for k in shared)
    for variant,n in [('C0',3),('C1',4)]:
        torch.manual_seed(88);module=GLR(16,variant)
        x=torch.randn(2,16,8,8,requires_grad=True)
        y,w=module.forward_with_weights(x)
        assert y.shape==x.shape and w.shape==(2,n,8,8)
        assert torch.allclose(w,torch.full_like(w,1/n)) and torch.allclose(w.sum(1),torch.ones_like(w[:,0]))
        assert module.gamma_c.item()>0 and module.gamma_s.item()>0
        assert torch.count_nonzero(module.refine(torch.zeros_like(x)))==0
        opt=torch.optim.SGD(module.parameters(),lr=.02)
        y.square().mean().backward()
        assert all(any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in branch.parameters()) for branch in module.branches)
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in module.protect.parameters())
        assert module.select_out.weight.grad is not None and module.select_out.weight.grad.abs().sum()>0
        opt.step();opt.zero_grad();module(torch.randn(2,16,8,8)).square().mean().backward()
        assert module.select_dw.weight.grad is not None and module.select_dw.weight.grad.abs().sum()>0
        assert all(torch.isfinite(p).all() for p in module.parameters())
        clone=GLR(16,variant);clone.load_state_dict(module.state_dict(),strict=True)
        assert all(isinstance(m,torch.nn.GroupNorm) and m.num_channels % m.num_groups==0 for m in module.modules() if isinstance(m,torch.nn.GroupNorm))
    for name in ('C0','C1'):
        out=D2GLR(name)(torch.randn(1,512,16,16));assert out.shape==(1,256,16,16)
    try: GLR(16,'C1')(torch.randn(1,16,3,3))
    except ValueError: pass
    else: raise AssertionError('invalid context resolution accepted')
    print('C0_C1_CPU_ACCEPTANCE_PASS; C2-C5/controls and split-FFN not implemented or tested',flush=True)
if __name__=='__main__':main()

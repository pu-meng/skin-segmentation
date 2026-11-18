import torch
from candidate import DirectionalWaveOperator

def test_directional():
 torch.manual_seed(42);torch.set_num_threads(1)
 op=DirectionalWaveOperator(dim=512,hidden_dim=512,res=8,num_heads=1)
 ax,ay=op.direction_coefficients()
 assert torch.equal(ax,torch.ones_like(ax)) and torch.equal(ay,torch.ones_like(ay))
 assert torch.equal(ax+ay,2*torch.ones_like(ax))
 assert not any(p.requires_grad for n,p in op.named_parameters() if n in ('t_raw','t_ch'))
 x=torch.randn(1,512,8,8,requires_grad=True);alpha=torch.full((8,5,1),-2.97,requires_grad=True)
 for step in range(3):
  op.zero_grad();y=op(x,alpha);assert y.shape==x.shape and torch.isfinite(y).all()
  y.square().mean().backward()
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in op.parameters() if p.requires_grad)
  with torch.no_grad():
   for p in op.parameters():
    if p.requires_grad:p.add_(p.grad,alpha=-.01)
 print('DIRECTIONAL_PASS',flush=True)
if __name__=='__main__':test_directional()

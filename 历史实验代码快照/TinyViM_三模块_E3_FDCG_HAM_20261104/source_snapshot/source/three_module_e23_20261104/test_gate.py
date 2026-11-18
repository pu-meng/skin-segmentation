import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from candidate import Gate
torch.set_num_threads(1)
g=Gate(3,5)
a=torch.randn(2,5,4,4);b=torch.randn(2,3,4,4)
assert torch.equal(g(a,b),a)
opt=torch.optim.SGD(g.parameters(),lr=.1)
seq=[]
for i in range(3):
 opt.zero_grad();g(a,b).square().mean().backward();seq.append(float(g.proj.weight.grad.norm()));opt.step()
assert seq[0]==0 and seq[1]>0 and seq[2]>0
assert g.raw.grad.abs().sum()>0
assert torch.tanh(g.raw).abs().max()<=1
print('GATE_PASS',seq)

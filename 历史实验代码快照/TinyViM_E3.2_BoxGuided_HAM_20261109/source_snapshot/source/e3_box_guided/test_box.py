import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util
from pathlib import Path
p=Path(__file__).with_name('candidate.py')
assert p.exists(), 'E3.2 implementation missing'
import torch
from candidate import legal_box,soft_rectangle,mask_boxes
b=legal_box(torch.zeros(3,4))
assert b.shape==(3,4)
assert torch.all(b[:,2:]>0)
w=soft_rectangle(b,64,64)
assert w.shape==(3,1,64,64) and torch.isfinite(w).all()
assert w[0,0,32,32]>w[0,0,0,0]
y=torch.zeros(3,1,256,256);y[1,0,0,0]=1;y[2,0,40:80,20:70]=1
boxes,valid=mask_boxes(y)
assert valid.tolist()==[False,True,True]
assert torch.allclose(boxes[1],torch.tensor([.5/256,.5/256,1/256,1/256]))
print('UNIT_PASS')

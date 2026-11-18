import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from common import environment
environment('')
import torch
from types import SimpleNamespace
from tta import SubsetPredictor
p=SubsetPredictor(views=1,device=torch.device('cpu'))
p.configuration_manager=SimpleNamespace(patch_size=[5,6,7])
p.network=torch.nn.Conv3d(1,3,1)
p.perform_everything_on_device=True
calls=[]
def failing(data,slicers,do_on_device=True):
    calls.append(do_on_device); raise RuntimeError('injected memory failure')
p._internal_predict_sliding_window_return_logits=failing
try: p.predict_sliding_window_return_logits(torch.zeros(1,5,6,7))
except RuntimeError: pass
else: raise AssertionError('error swallowed')
assert calls==[True],calls
print('PASS fail-fast sliding: one attempt, original error propagated, no CPU fallback')

import os,sys,importlib.util
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
p=Path(__file__).resolve().parents[2]/'source/recipe_r1_20261105/recipe.py'
assert p.exists(), 'missing unique unified recipe implementation'
spec=importlib.util.spec_from_file_location('recipe',p);r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
assert r.factor(0)==.01
assert abs(r.factor(r.W-1)-1)<1e-14 and abs(r.factor(r.W)-1)<1e-14
assert abs(r.factor(r.S-1)-.01)<1e-14
assert all(r.factor(t+1)<=r.factor(t) for t in range(r.W,r.S-1))
import torch,io
m=torch.nn.Module();m.encoder=torch.nn.Linear(2,2);m.align=torch.nn.Linear(2,2);m.decoder=torch.nn.Linear(2,1)
o=r.make_optimizer(m);s=r.UpdateSchedule(o)
ids=[id(p) for g in o.param_groups for p in g['params']]
assert len(ids)==len(set(ids))==len(list(m.parameters()))
for t in [0,r.W-1,r.W,r.S-1]:
 s.update=t;s.apply();assert abs(o.param_groups[1]['lr']/o.param_groups[0]['lr']-10)<1e-12
s.update=12345;state=s.state_dict();s.apply();expected=[g['lr'] for g in o.param_groups]
s2=r.UpdateSchedule(o);s2.load_state_dict(state);s2.apply();assert [g['lr'] for g in o.param_groups]==expected
ps=list(m.parameters())
for p in ps:p.grad=torch.ones_like(p)*100
n=r.clip(ps);assert n>12
assert torch.sqrt(sum(p.grad.square().sum() for p in ps))<=12.00001
ps[0].grad[0,0]=float('nan')
try:r.clip(ps)
except RuntimeError:pass
else:raise AssertionError('nonfinite clip accepted')
assert not torch.cuda.is_initialized()
print('PASS LR endpoints monotonic tail ratios restart exhaustive unique groups global_clip finite NaN CPU')

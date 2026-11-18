"""CPU contract for moving the unchanged E3.1 calculation to S1 or S2."""
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from source.e31_skip_positions_20261106.candidate import ROOT, build_candidate, DynamicSkip
from source.e31_skip_positions_20261106.selection import ARM, LEVEL, CHANNELS, RESOLUTION, RUN
import torch
from mamba_ssm.ops.selective_scan_interface import selective_scan_ref

class Reference:
    @staticmethod
    def apply(u,delta,A,B,C,D=None,delta_bias=None,delta_softplus=False,nrows=1):
        return selective_scan_ref(u,delta,A,B,C,D,z=None,delta_bias=delta_bias,delta_softplus=delta_softplus)

for name in ['Models.TinyViM.blocks','model.tvimblock']:
    importlib.import_module(name).SelectiveScan = Reference
torch.set_num_threads(4)
checks = {}
def ck(name, ok):
    assert ok, name
    checks[name] = True
    print(ARM, 'PASS', name, flush=True)

# Prove the generalized S0 is the existing E3.1, including RNG and tensor names.
from Models.E3DynamicSkip.model import build_candidate as original_build
from Models.E31SkipPositions.model import build_candidate as generalized_build
original = original_build(True).eval()
s0 = generalized_build(True,level=0).eval()
ck('S0_original_state_exact', original.state_dict().keys()==s0.state_dict().keys() and
   all(torch.equal(v,s0.state_dict()[k]) for k,v in original.state_dict().items()))
torch.manual_seed(42)
s = torch.randn(2,64,8,8);d = torch.randn_like(s)
with torch.no_grad():
    ck('S0_original_refiner_output_exact',torch.equal(original.decoder.refiner(s,d),s0.decoder.refiner(s,d)))
del original,s0

# Decoder dependence, official FDConv operation and real gradients at target size.
module = DynamicSkip(CHANNELS)
ck('official_FDW_KSM_FBM',hasattr(module.fdconv,'dft_weight') and
   all(hasattr(module.fdconv,n) for n in ['KSM_Global','KSM_Local','FBM']))
s = torch.randn(2,CHANNELS,8,8)
d = torch.randn_like(s,requires_grad=True)
ck('eta_initial_001',torch.equal(module.eta,torch.full_like(module.eta,.001)))
with torch.no_grad():
    w=module.select(d).softmax(1)
    ck('initial_weights_half',torch.equal(w,torch.full_like(w,.5)))
    ck('disabled_refiner_returns_S',torch.equal(module(s,d,enabled=False),s))
opt=torch.optim.AdamW(module.parameters(),lr=.001,weight_decay=.01)
before={n:p.detach().clone() for n,p in module.named_parameters()}
for step in range(3):
    module.train();opt.zero_grad(set_to_none=True);d.grad=None
    out=module(s,d)
    loss=(out-torch.randn_like(out)).square().mean()
    loss.backward()
    ck(f'finite_gradients_{step}',all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()))
    ck(f'D_receives_gradient_{step}',d.grad is not None and torch.isfinite(d.grad).all() and d.grad.norm()>0)
    if step==2:
        ck('all_added_gradients_nonzero',all(p.grad.count_nonzero()>0 for p in module.parameters()))
    opt.step()
ck('all_added_tensors_updated',all(not torch.equal(before[n],p.detach()) for n,p in module.named_parameters()))
module.eval()
with torch.no_grad():
    ck('different_D_changes_output',not torch.equal(module(s,d),module(s,torch.zeros_like(d))))
del module,opt,before

from Models.TinyViM.four_stage_decoder import build_four_stage_baseline
base=build_four_stage_baseline(True).eval();model=build_candidate(True).eval()
ck('all_retained_E0_tensors_exact',all(torch.equal(v,model.state_dict()[k]) for k,v in base.state_dict().items()))
ck('selected_level_only',model.decoder.level==LEVEL and sum(isinstance(m,DynamicSkip) for m in model.modules())==1)
model.decoder.enabled=False
x=torch.rand(1,3,256,256)
with torch.no_grad():ck('disabled_full_network_exact_E0',torch.equal(base(x),model(x)))
del base
model.decoder.enabled=True
seen={}
def capture(mod,args,out):
    seen.update(S=list(args[0].shape),D=list(args[1].shape),output=list(out.shape))
hook=model.decoder.refiner.register_forward_hook(capture)
with torch.no_grad():out=model(x)
hook.remove()
expected=[1,CHANNELS,RESOLUTION,RESOLUTION]
ck('actual_target_interfaces',seen==dict(S=expected,D=expected,output=expected))
ck('full_output_shape',list(out.shape)==[1,1,256,256] and torch.isfinite(out).all())
buf=io.BytesIO();torch.save(model.state_dict(),buf);buf.seek(0)
rebuilt=build_candidate(False).eval();rebuilt.load_state_dict(torch.load(buf,weights_only=True),strict=True)
with torch.no_grad():ck('independent_rebuild_strict_exact',torch.equal(out,rebuilt(x)))
ck('no_CUDA_initialized',not torch.cuda.is_initialized())
RUN.mkdir(parents=True,exist_ok=True)
record=dict(passed=True,arm=ARM,parameters=sum(p.numel() for p in model.parameters()),
            checks=checks,interfaces=seen,cpu_gradient_resolution=8,full_network_resolution=256,dataset_accessed=False,
            source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
with (RUN/'cpu_check.json').open('x') as f:json.dump(record,f,indent=2)
print('CPU_ACCEPTANCE_PASS',ARM,flush=True)

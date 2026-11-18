import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib
import importlib
import io
import json
from pathlib import Path
from source.direct_skip_gate_20261106.candidate import build_candidate,DirectSkipGate,ROOT
from source.direct_skip_gate_20261106.selection import RUN,ARM
import torch
from mamba_ssm.ops.selective_scan_interface import selective_scan_ref

class Reference:
    @staticmethod
    def apply(u,delta,A,B,C,D=None,delta_bias=None,delta_softplus=False,nrows=1):
        return selective_scan_ref(u,delta,A,B,C,D,z=None,delta_bias=delta_bias,delta_softplus=delta_softplus)
for name in ['Models.TinyViM.blocks','model.tvimblock']:
    importlib.import_module(name).SelectiveScan=Reference
torch.set_num_threads(4)
checks={}
def ck(name,ok):
    assert ok,name
    checks[name]=True
    print('PASS',name,flush=True)

gate=DirectSkipGate()
s=torch.randn(2,64,8,8);d=torch.randn_like(s,requires_grad=True)
ck('only_one_conv',sum(isinstance(m,torch.nn.Conv2d) for m in gate.modules())==1)
ck('gate_parameters_4160',sum(p.numel() for p in gate.parameters())==4160)
with torch.no_grad():
    ck('initial_weights_half',torch.equal(gate.weights(d),torch.full_like(d,.5)))
    ck('initial_output_half_skip',torch.equal(gate(s,d),.5*s))
    ck('zero_skip_zero_output',torch.equal(gate(torch.zeros_like(s),d),torch.zeros_like(s)))
    gate.proj.weight.copy_(torch.eye(64).reshape(64,64,1,1))
    ck('D_controls_gate',not torch.equal(gate(s,d),gate(s,torch.zeros_like(d))))
    ck('exact_elementwise_selection',torch.equal(gate(s,d),s*d.sigmoid()))
    gate.proj.weight.zero_()
opt=torch.optim.AdamW(gate.parameters(),lr=.001,weight_decay=.01)
before={n:p.detach().clone() for n,p in gate.named_parameters()}
for step in range(3):
    opt.zero_grad(set_to_none=True);d.grad=None
    out=gate(s,d);loss=(out-torch.randn_like(out)).square().mean();loss.backward()
    ck(f'gate_gradients_finite_nonzero_{step}',all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.count_nonzero()>0 for p in gate.parameters()))
    if step==2:ck('D_receives_gradient_after_learning',d.grad is not None and d.grad.norm()>0)
    opt.step()
ck('all_gate_parameters_updated',all(not torch.equal(v,dict(gate.named_parameters())[n]) for n,v in before.items()))
with torch.no_grad():ck('weights_between_zero_one',((gate.weights(d)>0)&(gate.weights(d)<1)).all())

from Models.TinyViM.four_stage_decoder import build_four_stage_baseline
base=build_four_stage_baseline(True).eval();model=build_candidate(True).eval()
ck('retained_E0_tensors_exact',all(torch.equal(v,model.state_dict()[k]) for k,v in base.state_dict().items()))
ck('only_extra_gate_4160',sum(p.numel() for p in model.parameters())-sum(p.numel() for p in base.parameters())==4160)
model.decoder.enabled=False;x=torch.rand(1,3,256,256)
with torch.no_grad():ck('disabled_exact_E0',torch.equal(base(x),model(x)))
del base
seen={}
def hook(mod,args,out):seen.update(S=list(args[0].shape),D=list(args[1].shape),output=list(out.shape))
handle=model.decoder.gate.register_forward_hook(hook)
model.decoder.enabled=True
with torch.no_grad():out=model(x)
handle.remove();ck('real_S0_interface',all(v==[1,64,64,64] for v in seen.values()) and len(seen)==3)
ck('output_shape_finite',list(out.shape)==[1,1,256,256] and torch.isfinite(out).all())
buf=io.BytesIO();torch.save(model.state_dict(),buf);buf.seek(0)
rebuilt=build_candidate(False).eval();rebuilt.load_state_dict(torch.load(buf,weights_only=True),strict=True)
with torch.no_grad():ck('independent_strict_reload_exact',torch.equal(out,rebuilt(x)))
ck('CPU_only_no_data',not torch.cuda.is_initialized())
RUN.mkdir(parents=True,exist_ok=True)
with (RUN/'cpu_check.json').open('x') as f:
    json.dump(dict(passed=True,arm=ARM,parameters=sum(p.numel() for p in model.parameters()),checks=checks,interfaces=seen,
                   dataset_accessed=False,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}),f,indent=2)
print('CPU_ACCEPTANCE_PASS',flush=True)

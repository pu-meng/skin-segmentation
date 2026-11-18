import os,sys,json,io,importlib,hashlib
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
import torch
from source.bd_controls_20261105.candidate import build_candidate,added_parameters,BasicMambaResidual
from source.bd_controls_20261105.selection import ARM,RUN,COUNT,MED
from Models.TinyViM.four_stage_decoder import build_four_stage_baseline
from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
class Reference:
 @staticmethod
 def apply(u,delta,A,B,C,D=None,delta_bias=None,delta_softplus=False,nrows=1):return selective_scan_ref(u,delta,A,B,C,D,z=None,delta_bias=delta_bias,delta_softplus=delta_softplus)
# Process-local official mathematical references; GPU launched in a fresh process.
for name in ['Models.TinyViM.blocks','model.tvimblock']:importlib.import_module(name).SelectiveScan=Reference
ms=importlib.import_module('mamba_ssm.modules.mamba_simple');ms.causal_conv1d_fn=None;ms.selective_scan_fn=selective_scan_ref
checks={}
def ck(k,v):assert v,k;checks[k]=True;print(ARM,'PASS',k,flush=True)
torch.set_num_threads(4);torch.manual_seed(42)
model=build_candidate(True);base=build_four_stage_baseline(True);off=build_candidate(True,count=0,med=False)
ck('all_off_state_exact_E0',all(torch.equal(v,off.state_dict()[k]) for k,v in base.state_dict().items()))
ck('retained_E0_tensors_exact',all(torch.equal(v,model.state_dict()[k]) for k,v in base.state_dict().items() if not (MED and k.startswith('decoder.stages.'))))
blocks=[m for m in model.modules() if isinstance(m,BasicMambaResidual)]
ck('official_Mamba_count',len(blocks)==COUNT)
ck('no_parameter_sharing',len({p.data_ptr() for b in blocks for p in b.parameters()})==sum(len(list(b.parameters())) for b in blocks))
x=torch.rand(1,3,256,256);base.eval();off.eval()
with torch.no_grad():ck('all_off_output_exact_E0',torch.equal(base(x),off(x)))
del base,off
model.train();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);before={n:p.detach().clone() for n,p in added_parameters(model)}
y=model(x);target=torch.rand_like(y);loss=torch.nn.functional.binary_cross_entropy(y,target)+1-(2*(y*target).sum()+1)/(y.sum()+target.sum()+1);loss.backward()
ck('real_backbone_output_shape',list(y.shape)==[1,1,256,256]);ck('all_gradients_finite',all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
ck('all_added_tensor_gradients_nonzero',all(torch.count_nonzero(p.grad)>0 for n,p in added_parameters(model)))
opt.step();ck('all_added_tensors_updated',all(not torch.equal(before[n],p.detach()) for n,p in added_parameters(model)))
model.eval()
with torch.no_grad():ref=model(x)
buf=io.BytesIO();torch.save(model.state_dict(),buf);buf.seek(0);model.load_state_dict(torch.load(buf,map_location='cpu',weights_only=True),strict=True)
with torch.no_grad():ck('strict_reload_exact',torch.equal(ref,model(x)))
ck('no_CUDA_initialized',not torch.cuda.is_initialized())
record=dict(source_sha256={str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},passed=True,arm=ARM,parameters=sum(p.numel() for p in model.parameters()),checks=checks,loss=float(loss.detach()),cpu_reference='official selective_scan_ref and official Mamba slow torch-conv path; process-local only',batch=1,shape=[1,3,256,256],dataset_accessed=False)
RUN.mkdir(parents=True,exist_ok=True)
with (RUN/'cpu_check.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record),flush=True)

import os,sys,json,io,importlib,hashlib
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
import torch
import importlib.util
assert importlib.util.find_spec('source.wpo_mr_20261105.candidate') is not None, 'missing isolated skip implementation'
from source.wpo_mr_20261105.candidate import build_candidate,added_parameters,ParallelBottleneck
from source.wpo_mr_20261105.selection import ARM,RUN
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
model=build_candidate(True);base=build_four_stage_baseline(True);off=build_candidate(True,enabled=False)
ck('all_off_state_exact_E0',all(torch.equal(v,off.state_dict()[k]) for k,v in base.state_dict().items()))
ck('retained_E0_tensors_exact',all(torch.equal(v,model.state_dict()[k]) for k,v in base.state_dict().items()))
blocks=[m for m in model.modules() if isinstance(m,ParallelBottleneck)]
ck('one_parallel_module',len(blocks)==1)
module=blocks[0];wave=module.a
ck('stage_parameters',list(wave.alpha_embed.shape)==[8,5,1] and list(wave.gamma.shape)==[1,512,1,1] and wave.wave.direction_raw.numel()==512)
ck('gamma_init',torch.equal(wave.gamma.detach(),torch.full_like(wave.gamma,.001)))
ck('fixed_time',not any(n in ['t_raw','t_ch'] for n,p in wave.wave.named_parameters()))
ck('static_c_1024_init1',module.c.numel()==1024 and tuple(module.c.shape)==(1,1024,1,1) and torch.equal(module.c,torch.ones_like(module.c)))
from source.wpo_mr_20261105.e1_wave import DStage,DirectionalWaveOperator
import ast
original=ast.parse((ROOT/'source/three_module_impl_20261104/candidate.py').read_text());local=ast.parse((Path(__file__).parent/'e1_wave.py').read_text())
ck('original_E1_class_AST_exact',all(ast.dump(next(v for v in original.body if isinstance(v,ast.ClassDef) and v.name==n))==ast.dump(next(v for v in local.body if isinstance(v,ast.ClassDef) and v.name==n)) for n in ['DStage','DirectionalWaveOperator']))
ck('original_official_wave_bytes_exact',(ROOT/'source/three_module_impl_20261104/official_wave.py').read_bytes()==(Path(__file__).parent/'official_wave.py').read_bytes())
ck('D3_retained_DoubleConv',model.decoder.stages[3].__class__.__name__=='DoubleConv')
seen={};hooks=[]
for name,sub in [('A',module.a),('B',module.b),('P',module.projection),('post',module.post),('D3',model.decoder.stages[3])]:
 hooks.append(sub.register_forward_hook(lambda m,i,o,n=name:seen.update({n:dict(input=list(i[0].shape),output=list(o.shape))})))
x=torch.rand(1,3,256,256);base.eval();off.eval()
with torch.no_grad():ck('all_off_output_exact_E0',torch.equal(base(x),off(x)))
del base,off
model.train();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);before={n:p.detach().clone() for n,p in added_parameters(model)}
y=model(x);target=torch.rand_like(y);loss=torch.nn.functional.binary_cross_entropy(y,target)+1-(2*(y*target).sum()+1)/(y.sum()+target.sum()+1);loss.backward()
ck('real_backbone_output_shape',list(y.shape)==[1,1,256,256]);ck('all_gradients_finite',all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
ck('all_added_tensor_gradients_present',all(p.grad is not None and torch.isfinite(p.grad).all() for n,p in added_parameters(model)))
opt.step()
for _ in range(2):
 opt.zero_grad(set_to_none=True);y=model(x);torch.nn.functional.binary_cross_entropy(y,target).backward();opt.step()
ck('all_added_tensors_updated',all(not torch.equal(before[n],p.detach()) for n,p in added_parameters(model)))
ck('all_1024_static_scalars_learn',bool(torch.all(module.c.detach()!=before['decoder.enhance.c'])))
ck('branch_and_D3_shapes',seen['A']['input']==[1,512,8,8] and seen['A']['output']==[1,512,8,8] and seen['B']['output']==[1,512,8,8] and seen['P']['input']==[1,1024,8,8] and seen['post']['output']==[1,512,8,8] and seen['D3']['input']==[1,512,8,8])
for hook in hooks:hook.remove()
# Explicit equation equivalence on rectangular-independent fixed 8x8 interface, no extra residual.
module.eval();probe=torch.rand(2,512,8,8)
with torch.no_grad():
 expected=module.post(probe+module.projection(module.c*torch.cat([module.a(probe),module.b(probe)],1)))
 ck('forward_equation_exact',torch.equal(module(probe),expected))
model.eval()
with torch.no_grad():ref=model(x)
buf=io.BytesIO();torch.save(model.state_dict(),buf);buf.seek(0);model.load_state_dict(torch.load(buf,map_location='cpu',weights_only=True),strict=True)
with torch.no_grad():ck('strict_reload_exact',torch.equal(ref,model(x)))
ck('no_CUDA_initialized',not torch.cuda.is_initialized())
record=dict(source_sha256={str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},passed=True,arm=ARM,parameters=sum(p.numel() for p in model.parameters()),checks=checks,interfaces=seen,static_c_numel=module.c.numel(),loss=float(loss.detach()),cpu_reference='official selective_scan_ref and official Mamba slow torch-conv path; process-local only',batch=1,shape=[1,3,256,256],dataset_accessed=False)
RUN.mkdir(parents=True,exist_ok=True)
with (RUN/'cpu_check.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record),flush=True)


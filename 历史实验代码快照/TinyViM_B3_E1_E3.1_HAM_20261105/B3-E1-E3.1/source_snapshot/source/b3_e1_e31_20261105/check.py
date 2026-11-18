import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,io,importlib,hashlib,ast
from pathlib import Path
from source.b3_e1_e31_20261105.candidate import ROOT,build_candidate,added_parameters,BasicMambaResidual,DStage,DynamicSkip
from source.b3_e1_e31_20261105.selection import ARM,RUN
import torch
from Models.TinyViM.four_stage_decoder import build_four_stage_baseline
from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
class Reference:
 @staticmethod
 def apply(u,delta,A,B,C,D=None,delta_bias=None,delta_softplus=False,nrows=1):return selective_scan_ref(u,delta,A,B,C,D,z=None,delta_bias=delta_bias,delta_softplus=delta_softplus)
for name in ['Models.TinyViM.blocks','model.tvimblock']:importlib.import_module(name).SelectiveScan=Reference
ms=importlib.import_module('mamba_ssm.modules.mamba_simple');ms.causal_conv1d_fn=None;ms.selective_scan_fn=selective_scan_ref
checks={}
def ck(k,v):
 assert v,k
 checks[k]=True;print(ARM,'PASS',k,flush=True)
torch.set_num_threads(2);torch.manual_seed(42)
local=ast.parse((Path(__file__).parent/'components.py').read_text())
for path,names in [('source/bd_controls_20261105/candidate.py',['BasicMambaResidual']),('source/three_module_impl_20261104/candidate.py',['DirectionalWaveOperator','DStage'])]:
 tree=ast.parse((ROOT/path).read_text())
 for n in names:ck(n+'_original_AST_exact',ast.dump(next(v for v in local.body if isinstance(v,ast.ClassDef) and v.name==n))==ast.dump(next(v for v in tree.body if isinstance(v,ast.ClassDef) and v.name==n)))
model=build_candidate(True);rng=torch.get_rng_state().clone();torch.manual_seed(42);base=build_four_stage_baseline(True);base_rng=torch.get_rng_state().clone();off=build_candidate(True,False)
ck('module_rng_preserved',torch.equal(rng,base_rng))
ck('all_off_exact_E0',all(torch.equal(v,off.state_dict()[k]) for k,v in base.state_dict().items()))
ck('retained_E0_exact',all(torch.equal(v,model.state_dict()[k]) for k,v in base.state_dict().items() if not k.startswith('decoder.stages.3.')))
ck('official_encoder_loaded',model.encoder.loading_audit['all_loaded_tensors_equal'])
# Independent RNG same class construction matches each component initial state.
for seed,target,make in [(44,model.decoder.enhance,lambda:torch.nn.Sequential(*[BasicMambaResidual() for _ in range(3)])),(42,model.decoder.stages[3],DStage),(42,model.decoder.refiner,DynamicSkip)]:
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(seed);ref=make()
 ck('independent_module_init_'+str(seed)+'_'+target.__class__.__name__,all(torch.equal(v,target.state_dict()[k]) for k,v in ref.state_dict().items()));del ref
ck('three_independent_Mamba',len(model.decoder.enhance)==3 and len({id(p) for m in model.decoder.enhance for p in m.parameters()})==sum(1 for m in model.decoder.enhance for p in m.parameters()))
ck('E1_replaces_D3',type(model.decoder.stages[3]) is DStage)
ck('original_E31_class',type(model.decoder.refiner) is DynamicSkip)
ck('FDConv_full',all(hasattr(model.decoder.refiner.fdconv,k) for k in ['dft_weight','KSM_Global','KSM_Local','FBM']))
x=torch.rand(1,3,256,256);base.eval();off.eval()
with torch.no_grad():ck('all_off_forward_exact_E0',torch.equal(base(x),off(x)))
del base,off
seen={};hooks=[]
for n,m in [('B3',model.decoder.enhance),('E1_D3',model.decoder.stages[3]),('D2',model.decoder.stages[2]),('D1',model.decoder.stages[1]),('E31',model.decoder.refiner),('D0',model.decoder.stages[0])]:
 hooks.append(m.register_forward_hook(lambda m,i,o,n=n:seen.update({n:dict(inputs=[list(t.shape) for t in i],output=list(o.shape))})))
# Full-resolution routing is checked with the real network in eval mode.
# FDConv global BatchNorm requires batch >=2 for training; component gradient
# checks use batch2 at 8x8 and real full batch8 is checked on GPU before launch.
model.eval()
with torch.no_grad():
 y=model(x)
 ck('full256_output_shape',list(y.shape)==[1,1,256,256] and torch.isfinite(y).all())
steps=[]
for label,component,channels,dual in [('B3',model.decoder.enhance,512,False),('E1',model.decoder.stages[3],512,False),('E31',model.decoder.refiner,64,True)]:
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(123);inp=torch.randn(2,channels,8,8);dec=torch.randn_like(inp,requires_grad=True)
 component.train();opt=torch.optim.AdamW(component.parameters(),lr=.001,weight_decay=.01)
 before={n:p.detach().clone() for n,p in component.named_parameters()}
 for step in range(3):
  opt.zero_grad(set_to_none=True);dec.grad=None
  out=component(inp,dec) if dual else component(inp)
  loss=(out-torch.randn_like(out)).square().mean();loss.backward()
  ck(label+'_finite_gradients_step'+str(step),all(p.grad is not None and torch.isfinite(p.grad).all() for p in component.parameters()))
  norms={n:float(p.grad.double().norm()) for n,p in component.named_parameters()}
  steps.append(dict(component=label,step=step+1,loss=float(loss.detach()),gradient_norms=norms))
  if step==2:ck(label+'_all_gradients_nonzero',all(v>0 for v in norms.values()))
  if dual:ck('E31_D_gradient_step'+str(step),dec.grad is not None and torch.isfinite(dec.grad).all() and dec.grad.norm()>0)
  opt.step();print('CPU_COMPONENT_STEP',label,step+1,float(loss.detach()),flush=True)
 ck(label+'_all_tensors_updated',all(not torch.equal(v,dict(component.named_parameters())[n]) for n,v in before.items()))
# Hooks used below must describe the full network, rather than tiny probes.
model.eval();seen.clear()
with torch.no_grad():model(x)
ck('B3_E1_order_interfaces',seen['B3']['inputs']==[[1,512,8,8]] and seen['E1_D3']['inputs']==[[1,512,8,8]])
ck('E31_S_D_before_D0',seen['E31']['inputs']==[[1,64,64,64],[1,64,64,64]] and seen['D0']['inputs']==[[1,128,64,64]])
for h in hooks:h.remove()
model.eval()
with torch.no_grad():expected=model(x)
buf=io.BytesIO();torch.save(model.state_dict(),buf);buf.seek(0);other=build_candidate(False);other.load_state_dict(torch.load(buf,map_location='cpu',weights_only=True),strict=True);other.eval()
with torch.no_grad():ck('independent_test_strict_rebuild_exact',torch.equal(expected,other(x)))
ck('CUDA_not_initialized',not torch.cuda.is_initialized())
record=dict(passed=True,arm=ARM,parameters=sum(p.numel() for p in model.parameters()),checks=checks,interfaces=seen,steps=steps,shape=[1,3,256,256],batch=1,component_training_batch=2,component_training_resolution=8,dataset_accessed=False,cpu_reference='installed official selective_scan_ref; CPU-only process-local references; no frozen files modified',gradient_scope='each original component batch2 three-step gradients and updates; full256 forward and strict rebuild; real batch8 full-backbone gradients checked later on GPU',source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
with (RUN/'cpu_check.json').open('x') as f:json.dump(record,f,indent=2)
print('CPU_COMPONENTS_AND_FULL_FORWARD_PASS',ARM,record['parameters'],flush=True)

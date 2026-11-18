import os,sys,json,io,importlib
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
import torch
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t');prepare_imports('tinyvim_s')
from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
class Reference:
 @staticmethod
 def apply(u,delta,A,B,C,D=None,delta_bias=None,delta_softplus=False,nrows=1):
  assert u.device.type=='cpu';return selective_scan_ref(u,delta,A,B,C,D,z=None,delta_bias=delta_bias,delta_softplus=delta_softplus)
for name in ['Models.TinyViM.blocks','model.tvimblock']:importlib.import_module(name).SelectiveScan=Reference
ms=importlib.import_module('mamba_ssm.modules.mamba_simple');ms.selective_scan_fn=selective_scan_ref;ms.causal_conv1d_fn=None
from source.e123_candidate_a_20261106.candidate import E2A,OneWayResidual,build_candidate,joint
from mamba_ssm.modules.mamba_simple import Mamba
out=ROOT/'log/TinyViM_E123_CandidateA_E2Spectral_B3x3_HAM100_20261106';checks={}
def check(k,v):
 assert bool(v),k
 checks[k]=True;print('PASS',k,flush=True)
def seed():
 import random,numpy as np
 random.seed(42);np.random.seed(42);torch.manual_seed(42)
torch.set_num_threads(4)
seed();old=joint.build_candidate(True);oldrng=torch.get_rng_state().clone()
seed();model=build_candidate(True);check('public_CPU_RNG_unchanged',torch.equal(oldrng,torch.get_rng_state()))
oldsd=old.state_dict();sd=model.state_dict();keys=[k for k in oldsd if not k.startswith('decoder.e2.')]
check('all_retained_tensors_exactly_paired',all(torch.equal(oldsd[k],sd[k]) for k in keys))
check('E1_exact_original_class',type(model.decoder.stages[3]) is joint.DStage)
check('E3_exact_joint_dual_source_class',type(model.decoder.e3) is joint.E3)
del old,oldsd,sd
new=model.decoder.e2
check('three_independent_one_way_Mamba128',sum(isinstance(m,Mamba) for m in new.modules())==3 and len({id(v.mixer) for v in new.spatial})==3)
check('official_scan_settings',all(v.mixer.d_model==128 and v.mixer.d_state==16 and v.mixer.d_conv==4 and v.mixer.expand==2 and v.mixer.dt_rank==8 for v in new.spatial))
for m in model.modules():
 if isinstance(m,Mamba):m.use_fast_path=False
x=torch.randn(2,128,8,8,requires_grad=True);stats=new.spectral_statistics(x)
check('zero_spectrum_finite',torch.isfinite(new.spectral_statistics(torch.zeros_like(x))).all())
check('statistics_gradient_finite',torch.isfinite(torch.autograd.grad(stats.sum(),x)[0]).all())
check('gain_initial_one',torch.equal(2*new.channel(stats).sigmoid(),torch.ones(2,128,1)))
check('fusion_initial_half',torch.equal(new.select(torch.randn(2,256)).reshape(2,2,128).softmax(1),torch.full((2,2,128),.5)))
f=torch.fft.fftfreq(8);mask=(f[:,None].abs()<=.25)&(f[None,:].abs()<=.25);check('symmetric_25_low_frequency_bins',mask.sum()==25 and torch.equal(mask,mask[(-torch.arange(8))%8][:,(-torch.arange(8))%8]))
z=torch.randn(1,512,8,8);g=new.gamma.detach().clone();new.gamma.data.zero_();check('gamma0_exact_interface_identity',torch.equal(new(z),z));new.gamma.data.copy_(g)
seen={}
handles=[]
for name,mod in [('E2',new),('E1',model.decoder.stages[3]),('E3',model.decoder.e3)]:
 def hook(m,args,y,name=name):seen[name]={'inputs':[list(v.shape) for v in args],'outputs':[list(v.shape) for v in y] if isinstance(y,tuple) else list(y.shape)}
 handles.append(mod.register_forward_hook(hook))
model.train();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0)
before={n:p.detach().clone() for n,p in new.named_parameters()};losses=[]
for step in range(3):
 opt.zero_grad(set_to_none=True);y=model(torch.rand(1,3,256,256));target=torch.rand_like(y);loss=torch.nn.functional.binary_cross_entropy(y,target)+1-(2*(y*target).sum()+1)/(y.sum()+target.sum()+1);loss.backward()
 check('all_model_gradients_finite_step'+str(step+1),all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
 opt.step();losses.append(loss.item());print('CPU_STEP',step+1,loss.item(),flush=True)
# Isolate E2 at its real 512x8x8 boundary: full-network synthetic loss has tiny gradients.
localopt=torch.optim.AdamW(new.parameters(),lr=.001,weight_decay=0)
for j in range(3):
 localopt.zero_grad(set_to_none=True);xx=torch.randn(2,512,8,8);ll=torch.nn.functional.mse_loss(new(xx),xx+torch.randn_like(xx));ll.backward();check('isolated_E2_finite_step'+str(j+1),all(p.grad is not None and torch.isfinite(p.grad).all() for p in new.parameters()));localopt.step()
print('UNCHANGED',[(n,float(p.grad.abs().max())) for n,p in new.named_parameters() if torch.equal(before[n],p.detach())],flush=True)
check('all_new_tensors_updated_without_weight_decay',all(not torch.equal(before[n],p.detach()) for n,p in new.named_parameters()))
check('last_step_new_gradients_nonzero',all(p.grad.abs().any() for p in new.parameters()))
check('real_backbone_output_shape',list(y.shape)==[1,1,256,256]);check('three_real_boundaries_hooked',set(seen)=={'E1','E2','E3'})
for h in handles:h.remove()
model.eval();x=torch.rand(1,3,256,256)
with torch.no_grad():expected=model(x)
buf=io.BytesIO();torch.save(model.state_dict(),buf);buf.seek(0);reloaded=build_candidate(False)
for m in reloaded.modules():
 if isinstance(m,Mamba):m.use_fast_path=False
reloaded.load_state_dict(torch.load(buf,map_location='cpu',weights_only=True),strict=True);reloaded.eval()
with torch.no_grad():check('fresh_constructor_strict_reload_exact',torch.equal(expected,reloaded(x)))
check('CUDA_never_initialized',not torch.cuda.is_initialized())
record=dict(passed=True,checks=checks,parameters=sum(p.numel() for p in model.parameters()),new_parameter_tensors=len(before),paired_retained_tensors=len(keys),boundaries=seen,losses=losses,CPU_reference_only=True)
(out/'cpu_acceptance.json').write_text(json.dumps(record,indent=2));print('CPU_ACCEPTANCE',json.dumps(record),flush=True)

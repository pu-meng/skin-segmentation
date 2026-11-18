import os,sys,json,io,importlib,random
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
import numpy as np
import torch
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t');prepare_imports('tinyvim_s')
from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
ms=importlib.import_module('mamba_ssm.modules.mamba_simple');ms.selective_scan_fn=selective_scan_ref;ms.causal_conv1d_fn=None
from source.decoder_dual_a_20261106.candidate import build_candidate,build_a,DualStage,joint,Mamba
RUN=Path('/media/8T/PuMengYu/experiments/skin_segmentation/DecoderDualA_SCREEN100of300_20261106')
checks={}
def ck(k,v):
 assert bool(v),k
 checks[k]=True;print('PASS',k,flush=True)
def seed():random.seed(42);np.random.seed(42);torch.manual_seed(42)
torch.set_num_threads(4)
seed();a=build_a(True);rng=torch.get_rng_state().clone()
seed();m=build_candidate(True);ck('retained_CPU_RNG_equal',torch.equal(rng,torch.get_rng_state()))
sd=a.state_dict();new=m.state_dict();keys=[k for k in sd if not k.startswith('decoder.stages.2.')]
ck('all_retained_state_tensors_equal',all(k in new and torch.equal(sd[k],new[k]) for k in keys))
ck('new_stage_only_state_replacement',set(new)-set(sd)=={k for k in new if k.startswith('decoder.stages.2.')})
ck('E1_E2A_E3_classes_preserved',type(m.decoder.stages[3]) is joint.DStage and type(m.decoder.e2) is type(a.decoder.e2) and type(m.decoder.e3) is joint.E3)
ck('hook_boundary_512_to_256',json.loads((ROOT/'operations/decoder_dual_a_20261106/boundary.json').read_text())==dict(input=[1,512,16,16],output=[1,256,16,16],old_class='DoubleConv'))
del a,sd,new
block=m.decoder.stages[2];block.mamba.mixer.use_fast_path=False
ck('official_Mamba1_config',isinstance(block.mamba.mixer,Mamba) and block.mamba.mixer.d_model==256 and block.mamba.mixer.d_state==16 and block.mamba.mixer.d_conv==4 and block.mamba.mixer.expand==2 and block.mamba.mixer.dt_rank==16)
ck('joint_Rep7_reused',type(block.rep[1]) is joint.Rep and block.rep[1].large[0].kernel_size==(7,7))
x=torch.randn(2,512,4,5,requires_grad=True);block.eval()
with torch.no_grad():
 u=block.project(x);expected=u+.1*block.raw_m.tanh()*block.mamba.correction(u)+.1*block.raw_r.tanh()*block.rep(u)
 ck('pure_correction_formula',torch.equal(block(x),expected))
 gm=block.raw_m.clone();gr=block.raw_r.clone();block.raw_m.zero_();block.raw_r.zero_();ck('zero_gain_is_projection_NOT_old_DoubleConv',torch.equal(block(x),u));block.raw_m.copy_(gm);block.raw_r.copy_(gr)
block.train();opt=torch.optim.AdamW(block.parameters(),lr=.001,weight_decay=0)
before={n:p.detach().clone() for n,p in block.named_parameters()};losses=[]
for step in range(3):
 opt.zero_grad(set_to_none=True);out=block(x);target=torch.randn_like(out);loss=torch.nn.functional.mse_loss(out,target);loss.backward()
 ck('all_new_gradients_present_finite_'+str(step+1),all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters()))
 ck('scalar_gains_nonzero_gradient_'+str(step+1),block.raw_m.grad.abs()>0 and block.raw_r.grad.abs()>0)
 opt.step();losses.append(loss.item())
ck('all_new_tensors_update_without_decay',all(not torch.equal(before[n],p) for n,p in block.named_parameters()))
ck('small_block_shape',out.shape==(2,256,4,5))
buf=io.BytesIO();torch.save(m.state_dict(),buf);buf.seek(0);r=build_candidate(False);r.load_state_dict(torch.load(buf,map_location='cpu',weights_only=True),strict=True)
ck('final_constructor_strict_state_roundtrip',all(torch.equal(v,r.state_dict()[k]) for k,v in m.state_dict().items()))
ck('CUDA_never_initialized',not torch.cuda.is_initialized())
record=dict(passed=True,checks=checks,parameters=sum(p.numel() for p in m.parameters()),retained_state_tensors=len(keys),new_parameter_tensors=len(before),small_block_losses=losses,CPU_scope='construction/paired tensors/strict reload/small block official selective_scan_ref backward; no full-network CPU backward',boundary=json.loads((ROOT/'operations/decoder_dual_a_20261106/boundary.json').read_text()))
(RUN/'cpu_acceptance.json').write_text(json.dumps(record,indent=2));print('CPU_ACCEPTANCE',json.dumps(record),flush=True)

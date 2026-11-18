"""Synthetic E2.1 checks only; does not construct datasets or start training."""
from pathlib import Path
import os,sys,json,copy,io,importlib
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison')]
MODE=sys.argv[1]
if MODE=='cpu':os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from torch import nn
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t');prepare_imports('tinyvim_s')
from Models.E2Simplified.model import build_candidate,RepDW7
from Models.TinyViM.four_stage_decoder import build_four_stage_baseline
from source.three_module_e23_20261104.candidate import Gate,FullScan
if MODE=='cpu':
 from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
 class Reference:
  @staticmethod
  def apply(u,delta,A,B,C,D=None,delta_bias=None,delta_softplus=False,nrows=1):
   assert u.device.type=='cpu'
   return selective_scan_ref(u,delta,A,B,C,D,z=None,delta_bias=delta_bias,delta_softplus=delta_softplus)
 for name in ['Models.TinyViM.blocks','model.tvimblock']:importlib.import_module(name).SelectiveScan=Reference
checks={}
def check(key,value):
 assert value,key
 checks[key]=True;print('PASS',key,flush=True)
torch.set_num_threads(4);torch.manual_seed(42);model=build_candidate(pretrained=True)
base=build_four_stage_baseline(pretrained=True)
check('all_original_E0_tensors_preserved',all(torch.equal(v,model.state_dict()[k]) for k,v in base.state_dict().items()))
check('no_attention_modules',all('attention' not in type(m).__name__.lower() for m in model.modules()))
check('three_four_direction_scans',sum(isinstance(m,FullScan) for m in model.decoder.enhance.modules())==3)
check('only_two_cross_frequency_gates',sum(isinstance(m,Gate) for m in model.decoder.enhance.modules())==2)
block=model.decoder.enhance
check('internal_width256_external_width512',block.pre[0].groups==512 and block.pre[1].out_channels==256 and block.post[0].groups==256 and block.post[1].out_channels==512)
check('final_depthwise_512_then_pointwise_256',block.mixer.fuse[0].groups==512 and block.mixer.fuse[1].out_channels==256)
check('high_branches_3_7_dilated3',block.mixer.high.branches[0].conv.c.kernel_size==(3,3) and block.mixer.high.branches[1].conv.c.kernel_size==(7,7) and block.mixer.high.branches[2][0].dilation==(2,2))
device=torch.device('cpu' if MODE=='cpu' else 'cuda:0');model=model.to(device).eval();base=base.to(device).eval();x=torch.rand(1,3,256,256,device=device)
extra=model.decoder.enhance;model.decoder.enhance=nn.Identity()
with torch.no_grad():check('bypass_matches_E0_exactly',torch.equal(model(x),base(x)))
model.decoder.enhance=extra;del base
for band_name in ['low','high']:
 band=getattr(model.decoder.enhance.mixer,band_name)
 check(band_name+'_residual_scale_initial_one',torch.equal(band.skip_scale,torch.ones_like(band.skip_scale)))
 projected=band.project(torch.randn(2,256,8,8,device=device))
 with torch.no_grad():
  values=[op(part)+band.skip_scale*part for op,part in zip(band.branches,projected.split(band.widths,dim=1))]
  observed=[]
  handle=band.fuse.register_forward_pre_hook(lambda m,inputs:observed.append(inputs[0]))
  original=band.project
  band.project=nn.Identity()
  band(projected)
  band.project=original;handle.remove()
  check(band_name+'_three_branch_residuals_match_formula',torch.equal(torch.cat(values,dim=1),observed[0]))
rep=RepDW7(7).to(device).eval();features=torch.randn(2,7,8,8,device=device)
with torch.no_grad():check('RepDW7_reparameterization_matches',torch.allclose(rep(features),copy.deepcopy(rep).fuse()(features),rtol=1e-5,atol=1e-5))
losses=[]
if MODE=='gpu':
 model.train();x=x.repeat(2,1,1,1);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
 before={n:p.detach().cpu().clone() for n,p in model.decoder.enhance.named_parameters()}
 for step in range(3):
  opt.zero_grad(set_to_none=True);y=model(x);target=torch.rand_like(y)
  loss=torch.nn.functional.binary_cross_entropy(y,target)+1-(2*(y*target).sum()+1)/(y.sum()+target.sum()+1)
  loss.backward();check('finite_gradients_'+str(step),all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
  opt.step();losses.append(float(loss))
 check('all_added_parameter_tensors_updated',all(not torch.equal(before[n],p.detach().cpu()) for n,p in model.decoder.enhance.named_parameters()))
model.eval()
with torch.no_grad():reference=model(x[:1]);check('output_shape',reference.shape==(1,1,256,256))
buffer=io.BytesIO();torch.save(model.state_dict(),buffer);buffer.seek(0)
other=build_candidate(pretrained=False).to(device).eval();other.load_state_dict(torch.load(buffer,map_location=device,weights_only=True),strict=True)
with torch.no_grad():check('strict_save_reload_exact',torch.equal(reference,other(x[:1])))
record=dict(passed=True,mode=MODE,checks=checks,parameters=sum(p.numel() for p in model.parameters()),added_parameters=sum(p.numel() for p in model.decoder.enhance.parameters()),synthetic_batch=2 if MODE=='gpu' else 1,optimizer_steps=len(losses),losses=losses,formal_training_started=False,dataset_accessed=False)
Path(__file__).with_name(MODE+'_check.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(record,ensure_ascii=False),flush=True)

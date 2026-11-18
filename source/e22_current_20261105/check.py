from pathlib import Path
import os,sys,json,io,importlib
MODE=sys.argv[1]
if MODE=='cpu':os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
import torch
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t');prepare_imports('tinyvim_s')
from candidate import build_candidate
from Models.TinyViM.four_stage_decoder import build_four_stage_baseline
from source.e123_joint_20261105.candidate import E2,E3,DStage,Scan
if MODE=='cpu':
 from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
 class Reference:
  @staticmethod
  def apply(u,delta,A,B,C,D=None,delta_bias=None,delta_softplus=False,nrows=1):return selective_scan_ref(u,delta,A,B,C,D,z=None,delta_bias=delta_bias,delta_softplus=delta_softplus)
 for name in ['Models.TinyViM.blocks','model.tvimblock']:importlib.import_module(name).SelectiveScan=Reference
checks={}
def ck(k,v):assert v,k;checks[k]=True;print('PASS',k,flush=True)
torch.set_num_threads(4);torch.manual_seed(42);model=build_candidate(True)
base=build_four_stage_baseline(True)
ck('all_E0_tensors_preserved',all(torch.equal(v,model.state_dict()[k]) for k,v in base.state_dict().items()))
ck('E1_E3_disabled',not any(isinstance(m,(E3,DStage)) for m in model.modules()))
ck('latest_E2_enabled',isinstance(model.decoder.enhance,E2));ck('two_width64_scans',sum(isinstance(m,Scan) for m in model.modules())==2)
device=torch.device('cpu' if MODE=='cpu' else 'cuda:0');model=model.to(device);base=base.to(device);x=torch.rand(1 if MODE=='cpu' else 2,3,256,256,device=device);model.eval();base.eval()
gamma=model.decoder.enhance.gamma.detach().clone()
with torch.no_grad():
 model.decoder.enhance.gamma.zero_();ck('gamma0_exact_E0_output',torch.equal(model(x),base(x)));model.decoder.enhance.gamma.copy_(gamma)
del base
opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);before={n:p.detach().cpu().clone() for n,p in model.decoder.enhance.named_parameters()};model.train();losses=[]
for i in range(1 if MODE=='cpu' else 3):
 opt.zero_grad(set_to_none=True);y=model(x);target=torch.rand_like(y);loss=torch.nn.functional.binary_cross_entropy(y,target)+1-(2*(y*target).sum()+1)/(y.sum()+target.sum()+1);loss.backward();ck('all_gradients_finite_step'+str(i+1),all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()));opt.step();losses.append(float(loss.detach()))
ck('output_shape',list(y.shape)==[x.shape[0],1,256,256])
if MODE=='gpu':ck('all_E2_tensors_updated',all(not torch.equal(before[n],p.detach().cpu()) for n,p in model.decoder.enhance.named_parameters()))
model.eval()
with torch.no_grad():reference=model(x[:1]).cpu()
buf=io.BytesIO();torch.save(model.state_dict(),buf);buf.seek(0);model.load_state_dict(torch.load(buf,map_location=device,weights_only=True),strict=True)
with torch.no_grad():ck('strict_reload_exact',torch.equal(reference,model(x[:1]).cpu()))
if MODE=='cpu':ck('no_CUDA_initialized',not torch.cuda.is_initialized())
record=dict(passed=True,mode=MODE,parameters=sum(p.numel() for p in model.parameters()),checks=checks,optimizer_steps=len(losses),losses=losses,synthetic_batch=x.shape[0],dataset_accessed=False)
p=Path(__file__).with_name(MODE+'_check.json')
with p.open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record),flush=True)

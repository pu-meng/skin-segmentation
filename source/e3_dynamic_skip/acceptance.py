import os,sys,io,json,importlib
from pathlib import Path
MODE=sys.argv[1]
if MODE=='cpu':os.environ['CUDA_VISIBLE_DEVICES']=''
import bootstrap
import torch
from Models.E3DynamicSkip.model import DynamicSkip,build_candidate
from Models.TinyViM.four_stage_decoder import build_four_stage_baseline
HERE=Path(__file__).parent
torch.set_num_threads(1 if MODE=='cpu' else 4)
report=dict(mode=MODE,checks={},gradients=[]);old=[]
def check(k,v):
 assert v,k
 report['checks'][k]=True;print('PASS',k,flush=True)
if MODE=='cpu':
 from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
 class Ref:
  @staticmethod
  def apply(u,delta,A,B,C,D=None,delta_bias=None,delta_softplus=False,nrows=1):
   assert u.device.type=='cpu'
   return selective_scan_ref(u,delta,A,B,C,D,z=None,delta_bias=delta_bias,delta_softplus=delta_softplus)
 for name in ['Models.TinyViM.blocks','model.tvimblock']:
  module=importlib.import_module(name);old.append((module,module.SelectiveScan));module.SelectiveScan=Ref
try:
 dev=torch.device('cpu' if MODE=='cpu' else 'cuda:0')
 if MODE=='cpu':
  import test_contract;test_contract.test_missing();check('contract_green',True)
  m=DynamicSkip().to(dev);s=torch.randn(2,64,8,8);d=torch.randn_like(s,requires_grad=True)
 else:
  m=build_candidate(True).to(dev);s=torch.rand(8,3,256,256,device=dev);d=None
 target=m if MODE=='cpu' else m.decoder.refiner
 fd=target.fdconv
 check('FDW_enabled_no_plain_weight',hasattr(fd,'dft_weight') and not hasattr(fd,'weight') and fd.kernel_num==4 and fd.groups==1)
 check('KSM_global_local_FBM_enabled',all(hasattr(fd,k) for k in ['KSM_Global','KSM_Local','FBM']))
 check('eta_nonzero_001',torch.equal(target.eta,torch.full_like(target.eta,.001)))
 check('selection_initial_half',torch.equal(target.select(torch.randn(2,64,8,8,device=dev)).softmax(1),torch.full((2,2,8,8),.5,device=dev)))
 hooks={};handles=[]
 for name in ['global_path','repdw','fdconv','g_global','g_local','select','decoder_project']:
  handles.append(getattr(target,name).register_forward_hook(lambda mod,i,o,n=name:hooks.__setitem__(n,tuple(o.shape))))
 opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01)
 if MODE!='cpu':torch.cuda.reset_peak_memory_stats()
 for step in range(3):
  m.train();opt.zero_grad(set_to_none=True);y=m(s,d) if MODE=='cpu' else m(s)
  check('shape_'+str(step),y.shape==(s.shape if MODE=='cpu' else (8,1,256,256)))
  loss=(y-torch.randn_like(y)).square().mean() if MODE=='cpu' else torch.nn.functional.binary_cross_entropy(y,torch.rand_like(y))
  loss.backward();check('all_finite_'+str(step),all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
  grads={n:float(p.grad.double().norm()) for n,p in target.named_parameters()};report['gradients'].append(grads)
  check('FDW_real_gradient_'+str(step),grads['fdconv.dft_weight']>0)
  if step==0:check('gate_projection_initially_blocked',all(getattr(target,n).proj.weight.grad.count_nonzero()==0 for n in ['g_global','g_local']))
  if step==2:check('all_added_gradient_nonzero',all(v>0 for v in grads.values()))
  opt.step();print('STEP',step,float(loss),flush=True)
 for h in handles:h.remove()
 report['hook_shapes']=hooks
 if MODE=='cpu':
  check('D_receives_gradient',d.grad is not None and d.grad.norm()>0)
  # CPU full network forward only: original CUDA scan temporarily official ref.
  torch.manual_seed(42);base=build_four_stage_baseline(True).eval();net=build_candidate(True).eval()
  check('all_E0_initial_tensors_exact',all(torch.equal(v,net.state_dict()[k]) for k,v in base.state_dict().items()))
  check('official_encoder_loaded',net.encoder.loading_audit['all_loaded_tensors_equal'])
  net.decoder.enabled=False;x=torch.rand(1,3,256,256)
  with torch.no_grad():check('disabled_exact_E0',torch.equal(net(x),base(x)))
  del base
  net.decoder.enabled=True
  with torch.no_grad():y=net(x);check('CPU_full_forward_shape',y.shape==(1,1,256,256))
  buf=io.BytesIO();torch.save(net.state_dict(),buf);buf.seek(0);other=build_candidate(False);other.load_state_dict(torch.load(buf,weights_only=True),strict=True);other.eval()
  with torch.no_grad():check('independent_test_rebuild_strict_exact',torch.equal(y,other(x)))
  check('CUDA_not_initialized',not torch.cuda.is_initialized())
 else:
  m.eval()
  with torch.no_grad():y=m(s)
  buf=io.BytesIO();torch.save(m.state_dict(),buf);buf.seek(0);other=build_candidate(False).to(dev);other.load_state_dict(torch.load(buf,map_location=dev,weights_only=True),strict=True);other.eval()
  with torch.no_grad():check('GPU_strict_roundtrip_exact',torch.equal(y,other(s)))
  report.update(peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
 report['passed']=True
finally:
 for mod,v in old:mod.SelectiveScan=v
report['scan_restored']=all(mod.SelectiveScan is v for mod,v in old)
(HERE/(MODE+'_acceptance.json')).write_text(json.dumps(report,indent=2))
print('ACCEPTANCE_PASS',MODE,flush=True)

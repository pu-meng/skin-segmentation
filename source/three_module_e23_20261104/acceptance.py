import os,sys,json,io,importlib,time
from pathlib import Path
MODE=sys.argv[1];ARM=sys.argv[2]
if MODE=='cpu':os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import torch
from candidate import build_candidate,ideal_split,skip_split,Gate
from pretrained_unet2d import prepare_imports
prepare_imports('tinyvim_s');prepare_imports('mambavision_t')
torch.set_num_threads(1 if MODE=='cpu' else 4)
D=Path(__file__).parent;out={'device':MODE,'arm':ARM,'checks':{},'gradient_steps':[]}
def check(k,v):
 assert v,k
 out['checks'][k]=True;print('PASS',k,flush=True)
orig=[]
if MODE=='cpu':
 from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
 class Reference:
  @staticmethod
  def apply(u,delta,A,B,C,D=None,delta_bias=None,delta_softplus=False,nrows=1):
   assert u.device.type=='cpu'
   return selective_scan_ref(u,delta,A,B,C,D,z=None,delta_bias=delta_bias,delta_softplus=delta_softplus)
 for name in ['Models.TinyViM.blocks','model.tvimblock']:
  m=importlib.import_module(name);orig.append((m,m.SelectiveScan));m.SelectiveScan=Reference
try:
 device=torch.device('cpu' if MODE=='cpu' else 'cuda:0')
 torch.manual_seed(42);base=build_candidate(pretrained=True).eval();model=build_candidate(pretrained=True,arm=ARM).eval()
 check('official_pretrained_equal',model.encoder.loading_audit['all_loaded_tensors_equal'] and all(torch.equal(v,model.encoder.state_dict()[k]) for k,v in base.encoder.state_dict().items()))
 check('all_baseline_tensors_preserved',all(torch.equal(v,model.state_dict()[k]) for k,v in base.state_dict().items()))
 model=model.to(device);base=base.to(device);off=build_candidate(pretrained=False).to(device);off.load_state_dict(base.state_dict(),strict=True);off.eval()
 x=torch.rand(1,3,256,256,device=device)
 with torch.no_grad():check('all_off_bitwise',torch.equal(base(x),off(x)))
 del base,off
 for shape in [(8,8),(7,7),(7,8)]:
  t=torch.randn(1,3,*shape,device=device);lo,hi=ideal_split(t)
  check('fft_reconstruct_'+str(shape),torch.allclose(lo+hi,t,atol=1e-6))
  sp=torch.fft.rfft2(t,norm='ortho');splo=torch.fft.rfft2(lo,norm='ortho');sphi=torch.fft.rfft2(hi,norm='ortho')
  check('complement_conjugacy_'+str(shape),torch.allclose(splo+sphi,sp,atol=2e-5) and torch.allclose(splo*sphi.conj(),torch.zeros_like(sp),atol=2e-5))
 t=torch.randn(1,64,64,64,device=device);lo,hi=skip_split(t);check('skip_reconstruct',torch.allclose(lo+hi,t,atol=1e-6))
 model.train();x=x.repeat(1 if MODE=='cpu' else 8,1,1,1)
 if MODE!='cpu':torch.cuda.reset_peak_memory_stats()
 opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
 gates=[(n,m) for n,m in model.named_modules() if isinstance(m,Gate)]
 check('gate_zero_init',all(torch.count_nonzero(m.raw)==0 for n,m in gates))
 for step in range(3):
  opt.zero_grad(set_to_none=True);y=model(x);check('shape_'+str(step),tuple(y.shape)==(len(x),1,256,256))
  target=torch.rand_like(y);loss=torch.nn.functional.binary_cross_entropy(y,target)+1-(2*(y*target).sum()+1)/(y.sum()+target.sum()+1)
  loss.backward();check('all_grad_finite_'+str(step),all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad))
  new={n:float(p.grad.double().norm()) for n,p in model.named_parameters() if n.startswith('decoder.enhance.')}
  out['gradient_steps'].append(new)
  if step==0:check('gate_projection_blocked_at_zero',all(m.proj.weight.grad.count_nonzero()==0 for n,m in gates))
  if step>0:check('gate_projection_unblocked_'+str(step),all(m.proj.weight.grad.norm()>0 for n,m in gates))
  if step==2:check('all_new_grad_nonzero',all(v>0 for v in new.values()))
  opt.step();print('OPTIMIZER_STEP',step,float(loss),flush=True)
 model.eval()
 with torch.no_grad():ref=model(x)
 buf=io.BytesIO();torch.save(model.state_dict(),buf);buf.seek(0)
 other=build_candidate(pretrained=False,arm=ARM).to(device);other.load_state_dict(torch.load(buf,map_location=device,weights_only=True),strict=True);other.eval()
 with torch.no_grad():check('strict_roundtrip_bitwise',torch.equal(ref,other(x)))
 out['parameters']=sum(p.numel() for p in model.parameters())
 if MODE!='cpu':out.update(peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
finally:
 for m,v in orig:m.SelectiveScan=v
out['scan_restored']=all(m.SelectiveScan is v for m,v in orig);out['cuda_initialized']=torch.cuda.is_initialized();out['passed']=True
(D/(MODE+'_'+ARM+'_acceptance.json')).write_text(json.dumps(out,indent=2))
print('ACCEPTANCE_PASS',MODE,ARM,len(out['checks']),flush=True)

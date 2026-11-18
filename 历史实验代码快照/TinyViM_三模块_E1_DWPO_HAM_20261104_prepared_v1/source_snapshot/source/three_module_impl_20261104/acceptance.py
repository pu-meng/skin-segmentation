import os,sys,json,io,time,importlib,hashlib
from pathlib import Path
MODE=sys.argv[1]
if MODE=='cpu':os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import torch
from candidate import build_candidate,DirectionalWaveOperator
from pretrained_unet2d import prepare_imports
prepare_imports('tinyvim_s');prepare_imports('mambavision_t')
torch.set_num_threads(1 if MODE=='cpu' else 4)
D=Path(__file__).parent
out={'device':MODE,'checks':{},'torch':torch.__version__}
def check(k,v):
 assert v,k
 out['checks'][k]=True;print('PASS',k,flush=True)
original=[]
if MODE=='cpu':
 from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
 class CPUReferenceScan:
  @staticmethod
  def apply(u,delta,A,B,C,D=None,delta_bias=None,delta_softplus=False,nrows=1):
   assert u.device.type=='cpu'
   return selective_scan_ref(u,delta,A,B,C,D,z=None,delta_bias=delta_bias,delta_softplus=delta_softplus)
 for name in ['Models.TinyViM.blocks','model.tvimblock']:
  m=importlib.import_module(name);original.append((m,m.SelectiveScan));m.SelectiveScan=CPUReferenceScan
try:
 torch.manual_seed(42)
 base=build_candidate(pretrained=True).eval();model=build_candidate(pretrained=True,d=True).eval()
 check('encoder_all_tensors_equal',all(torch.equal(v,model.encoder.state_dict()[k]) for k,v in base.encoder.state_dict().items()))
 out['loading_audit']=model.encoder.loading_audit
 check('encoder_official_loaded',out['loading_audit']['all_loaded_tensors_equal'])
 check('unmodified_tensors_equal',all(torch.equal(v,model.state_dict()[k]) for k,v in base.state_dict().items() if not k.startswith('decoder.stages.3.')))
 device=torch.device("cpu" if MODE=="cpu" else "cuda:0");model=model.to(device);base=base.to(device)
 x=torch.rand(1,3,256,256,device=device)
 off=build_candidate(pretrained=False);off.load_state_dict(base.state_dict(),strict=True);off=off.to(device);off.eval()
 with torch.no_grad():check('all_off_bitwise',torch.equal(base(x),off(x)))
 del base,off
 device=torch.device('cpu' if MODE=='cpu' else 'cuda:0');model=model.to(device);model.train();x=x.to(device)
 if MODE!='cpu':
  torch.cuda.reset_peak_memory_stats();x=x.repeat(8,1,1,1)
 optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
 losses=[];grads={}
 for step in range(3):
  optimizer.zero_grad(set_to_none=True);y=model(x);check('full_shape_step'+str(step),tuple(y.shape)==(x.shape[0],1,256,256))
  target=torch.rand_like(y);loss=torch.nn.functional.binary_cross_entropy(y,target)+1-(2*(y*target).sum()+1)/(y.sum()+target.sum()+1)
  loss.backward();check('loss_finite_'+str(step),bool(torch.isfinite(loss)))
  check('all_trainable_grad_finite_'+str(step),all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad))
  grads={n:float(p.grad.double().norm()) for n,p in model.named_parameters() if n.startswith('decoder.stages.3.') and p.requires_grad}
  optimizer.step();losses.append(float(loss));print('OPTIMIZER_STEP',step,losses[-1],flush=True)
 check('all_new_tensor_grad_nonzero_step3',all(v>0 for v in grads.values()));out['new_parameter_gradient_norms']=grads;out['losses']=losses
 model.eval()
 with torch.no_grad():reference=model(x)
 buf=io.BytesIO();torch.save(model.state_dict(),buf);buf.seek(0)
 other=build_candidate(pretrained=False,d=True).to(device);other.load_state_dict(torch.load(buf,map_location=device,weights_only=True),strict=True);other.eval()
 with torch.no_grad():check('strict_roundtrip_bitwise',torch.equal(reference,other(x)))
 del other,optimizer
 op=model.decoder.stages[3].wave
 # Capture propagated spectra and independently verify rFFT validity.
 old=op._gate_and_project_out;specs=[]
 def capture(u,z,shape):
  specs.append(u.detach());return old(u,z,shape)
 op._gate_and_project_out=capture
 with torch.no_grad():
  for dims in [(8,8),(7,7),(7,8)]:
   xx=torch.randn(1,512,*dims,device=device);a=torch.randn(dims[0],dims[1]//2+1,1,device=device)
   yy=op(xx,a);sp=specs[-1];re=torch.fft.rfft2(torch.fft.irfft2(sp,s=dims,norm='ortho'),norm='ortho')
   check('conjugacy_'+str(dims),torch.allclose(re,sp,atol=2e-5,rtol=2e-5))
  check('zero_input_finite',bool(torch.isfinite(op(torch.zeros(1,512,8,8,device=device),torch.zeros(8,5,1,device=device))).all()))
 op._gate_and_project_out=old
 if MODE!='cpu':
  out['fp32_peak_allocated_bytes']=torch.cuda.max_memory_allocated();out['fp32_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
  model.train();model.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.float16):
   y=model(x);loss=y.float().square().mean()
  loss.backward();check('amp_finite',bool(torch.isfinite(loss)) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad))
  model.eval();model.zero_grad(set_to_none=True)
  with torch.no_grad():
   for _ in range(3):model(x)
   torch.cuda.synchronize();start=time.monotonic()
   for _ in range(10):model(x)
   torch.cuda.synchronize();out['latency_ms_batch8']=(time.monotonic()-start)*100
 out['parameters']=sum(p.numel() for p in model.parameters());out['flops']='not measured; FFT/complex/scan not counted'
finally:
 for m,orig in original:m.SelectiveScan=orig
out['cuda_initialized']=torch.cuda.is_initialized();out['passed']=True
(D/(MODE+'_acceptance.json')).write_text(json.dumps(out,ensure_ascii=False,indent=2))
print('ACCEPTANCE_PASS',MODE,len(out['checks']),flush=True)

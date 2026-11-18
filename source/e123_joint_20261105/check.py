from pathlib import Path
import os,sys,json,io,importlib,copy
MODE=sys.argv[1]
if MODE=='cpu':os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
import torch
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t');prepare_imports('tinyvim_s')
from candidate import *
from selection import RUN,CHOICES
HERE=Path(__file__).parent
if MODE=='cpu':
 from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
 class Reference:
  @staticmethod
  def apply(u,delta,A,B,C,D=None,delta_bias=None,delta_softplus=False,nrows=1):
   assert u.device.type=='cpu';return selective_scan_ref(u,delta,A,B,C,D,z=None,delta_bias=delta_bias,delta_softplus=delta_softplus)
 for name in ['Models.TinyViM.blocks','model.tvimblock']:importlib.import_module(name).SelectiveScan=Reference
checks={}
def check(k,v):
 assert v,k
 checks[k]=True;print('PASS',k,flush=True)
torch.set_num_threads(4);torch.manual_seed(42);device=torch.device('cpu' if MODE=='cpu' else 'cuda:0')
# Mathematical checks on the new operators, independent from backbone output shape.
x=torch.randn(2,24,10,14)
check('Haar_exact_analysis_synthesis',torch.allclose(Haar.synthesis(Haar.analysis(x)),x,atol=1e-6,rtol=1e-6))
f=FourierDyn();x=torch.randn(2,32,8,8,requires_grad=True);k=f.kernels(x)
y=f(x);ref=F.conv2d(x.reshape(1,64,8,8),k.flip((-2,-1)).reshape(64,1,3,3),padding=1,groups=64).reshape_as(x)
check('Fourier_linear_same_convolution',torch.allclose(y,ref,atol=1e-6,rtol=1e-5));gy=torch.autograd.grad(y.square().sum(),x,retain_graph=True)[0];gr=torch.autograd.grad(ref.square().sum(),x)[0];check('Fourier_gradient_matches',torch.allclose(gy,gr,atol=1e-5,rtol=1e-5))
# Explicit four individual selective_scan_refs independently verify group order and average, rectangular layout.
from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
sc=Scan().to(device);x=torch.randn(1,64,4,6,device=device,requires_grad=True)
if MODE=='gpu':torch.cuda.reset_peak_memory_stats()
y=sc(x);seq=[x.flatten(2),x.transpose(2,3).flatten(2)];seq+=[t.flip(-1) for t in seq];outs=[]
for i,t in enumerate(seq):
 xp=torch.einsum('bdl,cd->bcl',t,sc.xw[i]);dt,bb,cc=xp.split([4,16,16],1);dt=torch.einsum('brl,dr->bdl',dt,sc.dtw[i])
 z=selective_scan_ref(t,dt,-sc.A[i*64:(i+1)*64].exp(),bb,cc,sc.D[i*64:(i+1)*64],delta_bias=sc.dtb[i],delta_softplus=True)
 if i>=2:z=z.flip(-1)
 if i%2:z=z.reshape(1,64,6,4).transpose(2,3).reshape(1,64,24)
 outs.append(z)
ref=torch.stack(outs).mean(0).reshape_as(x);check('four_direction_average_rectangular',torch.allclose(y,ref,atol=2e-5,rtol=2e-4))
g1=torch.autograd.grad(y.square().mean(),x,retain_graph=True)[0];g2=torch.autograd.grad(ref.square().mean(),x)[0];check('scan_reference_gradient',torch.allclose(g1,g2,atol=2e-5,rtol=2e-4))
del sc,x,y,ref,outs
model=build_candidate(pretrained=True).to(device);model.eval();params=sum(p.numel() for p in model.parameters())
check('E1_original_DStage',isinstance(model.decoder.stages[3],DStage))
check('joint_E2_E3_enabled',isinstance(model.decoder.e2,E2) and isinstance(model.decoder.e3,E3))
check('two_independent_width64_scans',sum(isinstance(m,Scan) for m in model.modules())==2)
check('rhoG_rhoL_point1',abs(model.decoder.e2.mixer.rg.sigmoid().item()-.1)<1e-6 and abs(model.decoder.e2.mixer.rl.sigmoid().item()-.1)<1e-6)
check('outer_gamma_point01',all(torch.allclose(p,torch.full_like(p,.01)) for p in [model.decoder.e2.gamma,model.decoder.e3.s.gamma,model.decoder.e3.d.gamma]))
seen={}
handles=[]
for key,mod in [('E2',model.decoder.e2),('E1',model.decoder.stages[3]),('E3',model.decoder.e3)]:
 def hook(m,args,out,key=key):seen[key]=dict(inputs=[list(t.shape) for t in args],outputs=[list(t.shape) for t in out] if isinstance(out,tuple) else list(out.shape))
 handles.append(mod.register_forward_hook(hook))
# Full real backbone CPU forward/backward using exact official reference only on CPU.
model.train();x=torch.rand(1 if MODE=='cpu' else 2,3,256,256,device=device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
steps=1 if MODE=='cpu' else 3
before={n:p.detach().cpu().clone() for n,p in model.decoder.named_parameters()};losses=[];grad_zero=[]
for i in range(steps):
 opt.zero_grad(set_to_none=True);y=model(x);target=torch.rand_like(y);loss=F.binary_cross_entropy(y,target)+1-(2*(y*target).sum()+1)/(y.sum()+target.sum()+1);loss.backward()
 check('joint_finite_all_gradients_step'+str(i+1),all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
 grad_zero=[n for n,p in model.decoder.named_parameters() if not p.grad.abs().any()]
 opt.step();losses.append(float(loss.detach()));print('STEP',i+1,losses[-1],flush=True)
check('joint_output_shape',list(y.shape)==[x.shape[0],1,256,256])
check('three_real_boundaries_hooked',set(seen)=={'E1','E2','E3'})
if MODE=='gpu':check('all_decoder_parameter_tensors_updated_three_steps',all(not torch.equal(before[n],p.detach().cpu()) for n,p in model.decoder.named_parameters()))
for h in handles:h.remove()
model.eval()
with torch.no_grad():y=model(x[:1]).cpu()
buffer=io.BytesIO();torch.save(model.state_dict(),buffer);buffer.seek(0);model.load_state_dict(torch.load(buffer,map_location=device,weights_only=True),strict=True)
with torch.no_grad():check('strict_reload_exact',torch.equal(y,model(x[:1]).cpu()))
if MODE=='cpu':check('CUDA_never_initialized',not torch.cuda.is_initialized())
record=dict(passed=True,mode=MODE,parameters=params,checks=checks,optimizer_steps=steps,losses=losses,zero_gradient_tensors_last_step=grad_zero,boundaries=seen,choices=CHOICES,synthetic_batch=x.shape[0],formal_batch8_pending=True,dataset_accessed=False,cuda_initialized=torch.cuda.is_initialized())
if MODE=='gpu':record.update(peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
with (HERE/(MODE+'_check.json')).open('x') as f:json.dump(record,f,ensure_ascii=False,indent=2)
print('JOINT_ACCEPTANCE',json.dumps(record,ensure_ascii=False),flush=True)

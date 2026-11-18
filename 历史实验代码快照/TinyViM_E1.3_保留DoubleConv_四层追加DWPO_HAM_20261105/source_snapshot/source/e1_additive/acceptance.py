"""Check retained DoubleConv initialization, stage routing, updates and reload."""
import os,sys,json,io,importlib
from pathlib import Path
MODE=sys.argv[1]
if MODE=='cpu':os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from torch import nn
from candidate import ROOT,build_candidate
from selection import ARM
from pretrained_unet2d import prepare_imports
prepare_imports('tinyvim_s');prepare_imports('mambavision_t')
from Models.TinyViM.four_stage_decoder import build_four_stage_baseline
from Models.TinyViM.model import DoubleConv

torch.set_num_threads(2);torch.manual_seed(42)
checks={}
def check(name,value):
 assert value,name
 checks[name]=True;print('PASS',name,flush=True)
if MODE=='cpu':
 from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
 class Reference:
  @staticmethod
  def apply(u,delta,A,B,C,D=None,delta_bias=None,delta_softplus=False,nrows=1):
   assert u.device.type=='cpu'
   return selective_scan_ref(u,delta,A,B,C,D,z=None,delta_bias=delta_bias,delta_softplus=delta_softplus)
 for name in ('Models.TinyViM.blocks','model.tvimblock'):
  importlib.import_module(name).SelectiveScan=Reference
model=build_candidate(pretrained=True);base=build_four_stage_baseline(pretrained=True)
check('official_encoder_loaded',model.encoder.loading_audit['all_loaded_tensors_equal'])
state=model.state_dict()
for key,value in base.state_dict().items():
 if key.startswith('decoder.stages.'):
  bits=key.split('.');key='.'.join(bits[:3]+['0']+bits[3:])
 assert torch.equal(value,state[key]),key
check('all_original_encoder_alignment_doubleconv_up_head_tensors_equal',True)
check('four_original_doubleconv_retained',all(isinstance(stage[0],DoubleConv) for stage in model.decoder.stages))
check('four_additional_wave_blocks',len(model.decoder.stages)==4 and all(hasattr(stage[1],'wave') for stage in model.decoder.stages))
device=torch.device('cpu' if MODE=='cpu' else 'cuda:0');model=model.to(device).eval();base=base.to(device).eval()
x=torch.rand(1,3,256,256,device=device)
extras=[stage[1] for stage in model.decoder.stages]
for stage in model.decoder.stages:stage[1]=nn.Identity()
with torch.no_grad():check('bypass_all_additions_matches_baseline_exactly',torch.equal(model(x),base(x)))
for stage,extra in zip(model.decoder.stages,extras):stage[1]=extra
trace=[];handles=[]
for idx,stage in enumerate(model.decoder.stages):
 for pos in [0,1]:
  handles.append(stage[pos].register_forward_hook(lambda m,inp,out,i=idx,j=pos:trace.append((i,j,list(out.shape[1:])))))
with torch.no_grad():model(x)
check('doubleconv_then_addition_in_D3_D2_D1_D0',[(i,j) for i,j,shape in trace]==[(3,0),(3,1),(2,0),(2,1),(1,0),(1,1),(0,0),(0,1)])
check('stage_shapes_preserved',all(shape==[64*2**i,64//2**i,64//2**i] for i,j,shape in trace))
for h in handles:h.remove()
del base,state
model.train();batch=1 if MODE=='cpu' else 8;x=x.repeat(batch,1,1,1)
opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
added={n:p.detach().cpu().clone() for n,p in model.named_parameters() if n.startswith('decoder.stages.') and n.split('.')[3]=='1'}
if MODE!='cpu':torch.cuda.reset_peak_memory_stats()
losses=[]
for step in range(3):
 opt.zero_grad(set_to_none=True);y=model(x);target=torch.rand_like(y)
 loss=torch.nn.functional.binary_cross_entropy(y,target)+1-(2*(y*target).sum()+1)/(y.sum()+target.sum()+1)
 loss.backward()
 check('finite_output_loss_gradients_'+str(step),y.shape==(batch,1,256,256) and bool(torch.isfinite(loss)) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad))
 check('all_added_tensors_receive_nonzero_gradients_'+str(step),all(p.grad.abs().sum()>0 for n,p in model.named_parameters() if n in added))
 opt.step();losses.append(float(loss))
check('all_added_parameter_tensors_updated',all(not torch.equal(added[n],p.detach().cpu()) for n,p in model.named_parameters() if n in added))
model.eval()
with torch.no_grad():expected=model(x[:1])
buffer=io.BytesIO();torch.save(model.state_dict(),buffer);buffer.seek(0)
other=build_candidate(pretrained=False).to(device).eval();other.load_state_dict(torch.load(buffer,map_location=device,weights_only=True),strict=True)
with torch.no_grad():check('strict_checkpoint_roundtrip',torch.equal(expected,other(x[:1])))
record=dict(passed=True,arm=ARM,device=MODE,checks=checks,batch=batch,losses=losses,parameters=sum(p.numel() for p in model.parameters()),added_parameter_tensors=len(added),test_data_accessed=False)
if MODE!='cpu':record.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),physical_gpu=os.environ['EXPERIMENT_PHYSICAL_GPU'])
Path(__file__).with_name(ARM+'_'+MODE+'_acceptance.json').write_text(json.dumps(record,indent=2)+'\n')
print('ACCEPTANCE_PASS',ARM,MODE,flush=True)

import os,sys,json,copy,hashlib
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation');OPS=ROOT/'operations/duala_internal_bypass_20261107';OUT=Path('/media/8T/PuMengYu/experiments/skin_segmentation/DualA_InternalBypass_Prepare_20261107')
sys.path[:0]=[str(OPS),str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
from source.e1232_fourarm_screen_20261106 import entry as entry
import torch
from internal_bypass import build,InternalBypassE2,OLD
from candidates import build as old_build

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert not OUT.exists();OUT.mkdir();r={'status':'preparing','training_started':False,'scope':'CPU retained tensors + actual isolated E2 CUDA batch8 synthetic forward/backward/bypass; not wholemodel train acceptance'}
 def save():(OUT/'acceptance.json').write_text(json.dumps(r,ensure_ascii=False,indent=2))
 protected={str(OLD/n):sha(OLD/n) for n in ['candidates.py','entry.py','official_afno2d.py']};torch.set_num_threads(2);api=entry.runner.baseline_api()
 try:
  api.seed_all(42);base=old_build('Input',False);bs=base.state_dict();api.seed_all(42);new=build(False);ns=new.state_dict();extra=set(ns)-set(bs);assert extra=={'decoder.e2.depth_scale','decoder.e2.point_scale','decoder.e2.global_scale'};assert all(torch.equal(v,ns[k]) for k,v in bs.items());new.load_state_dict(copy.deepcopy(ns),strict=True)
  r['retained_tensors_equal']=len(bs);r['new_scale_parameters']=sum(ns[k].numel() for k in extra);r['parameters']=sum(p.numel() for p in new.parameters());r['encoder_trainable']=all(p.requires_grad for p in new.encoder.parameters());assert not torch.cuda.is_initialized();save()
  torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.05);e=copy.deepcopy(new.decoder.e2).cuda().train();x=torch.randn(8,512,8,8,device='cuda',requires_grad=True)
  scales={n:getattr(e,n).detach().clone() for n in ['depth_scale','point_scale','global_scale']}
  with torch.no_grad():
   z=e.pre(x);a,b=z.chunk(2,1)
   e.depth_scale.zero_();e.point_scale.zero_();e.global_scale.zero_();l,g=e.contents(z);assert torch.equal(l,a) and torch.equal(g,b)
   e.point_scale.copy_(scales['point_scale']);actual=e.local_contents(a);expect=a+e.point_scale*e.local[3](e.local[2](a));assert torch.equal(actual,expect)
   e.depth_scale.copy_(scales['depth_scale']);e.point_scale.zero_();actual=e.local_contents(a);expect=a+e.depth_scale*e.local[1](e.local[0](a));assert torch.equal(actual,expect)
   for n,v in scales.items():getattr(e,n).copy_(v)
   gamma=e.gamma.detach().clone();e.gamma.zero_();assert torch.equal(e(x),x);e.gamma.copy_(gamma)
  e.zero_grad(set_to_none=True);opt=torch.optim.AdamW(e.parameters(),lr=.001,weight_decay=0);before={n:getattr(e,n).detach().clone() for n in scales};y=e(x);target=torch.randn_like(y);loss=(y-target).square().mean();loss.backward();assert torch.isfinite(x.grad).all();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in e.parameters());r['scale_gradient_norms']={n:float(getattr(e,n).grad.norm()) for n in scales};assert all(v>0 for v in r['scale_gradient_norms'].values());opt.step();r['scale_updates']={n:float((getattr(e,n)-v).abs().max()) for n,v in before.items()};assert all(v>0 for v in r['scale_updates'].values())
  r.update(status='prepared',isolated_E2_cuda_batch8_passed=True,local_A_bypass_then_B_exact=True,local_B_bypass_after_A_exact=True,both_local_blocks_identity_exact=True,global_bypass_identity_exact=True,outer_gamma0_identity_exact=True,strict_load=True,finite_gradients=True,output_shape=list(y.shape),gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],source_unchanged=all(sha(p)==h for p,h in protected.items()),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
 except BaseException as err:r.update(status='failed',error=repr(err));save();raise
 save();(OUT/'protocol.json').write_text(json.dumps({'name':'DualA｜输入选路·内部逐段旁路','reference':'DualA｜局部全局·输入选路 (Input)','new_parameters':'3 static per-channel scales, each128; init0.1; not image-conditioned gates','equations':['h=a+alpha*DW7(LN(a))','L=h+beta*PW(GELU(h))','G=b+delta*MambaCorrection(GlobalNorm(b))','Y=X+gamma*Output(InputSelectedConcat(L,G))'],'shared_initialization':'all preexisting model tensors equal, only384 added scale parameters','frozen_original_sources':protected,'status':'prepared only, no training launched or queue mutation','implementation_sha256':{str(p):sha(p) for p in OPS.glob('*.py')}},ensure_ascii=False,indent=2));print('PREPARED',json.dumps(r,ensure_ascii=False),flush=True)
if __name__=='__main__':main()

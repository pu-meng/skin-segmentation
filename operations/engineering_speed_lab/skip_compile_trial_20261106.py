import os
os.environ.update(CUDA_VISIBLE_DEVICES='GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',TORCHINDUCTOR_COMPILE_THREADS='1')
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation');OUT=ROOT/'operations/engineering_speed_lab/skip_fusion_20261106';OUT.mkdir(exist_ok=True)
os.environ['TORCHINDUCTOR_CACHE_DIR']=str(OUT/'inductor_cache');os.environ['TRITON_CACHE_DIR']=str(OUT/'triton_cache')
import sys,json,time,copy,statistics,traceback
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
import torch
from source.e1232_fourarm_screen_20261106 import entry as entry
from source.e123_joint_20261105.candidate import E3Side
class Finish(torch.nn.Module):
 def __init__(self,side):super().__init__();self.side=side
 def forward(self,x,u,k):return self.side.finish(x,u,k)
def main():
 torch.set_num_threads(2);torch.cuda.set_device(0);torch.manual_seed(42)
 raw=Finish(E3Side()).cuda().train();candidate=copy.deepcopy(raw);compiled=torch.compile(candidate,fullgraph=True,mode='default')
 data=[torch.randn(8,64,64,64,device='cuda') for _ in range(3)];report={'scope':'one-side exact finish subgraph: mix projection+FFN+post+gamma+residual; training FP32 batch8 64x64; synthetic inputs; no wholemodel change','timings':[]}
 def save():(OUT/'result.json').write_text(json.dumps(report,indent=2))
 try:
  def step(mod,inputs):
   mod.zero_grad(set_to_none=True)
   for x in inputs:x.grad=None
   out=mod(*inputs);out.square().mean().backward();return out
  a=[v.detach().clone().requires_grad_() for v in data];b=[v.detach().clone().requires_grad_() for v in data]
  reference=step(raw,a);torch.cuda.synchronize();t=time.perf_counter();result=step(compiled,b);torch.cuda.synchronize();report['compile_and_first_step_seconds']=time.perf_counter()-t
  grads={n:p.grad for n,p in raw.named_parameters() if p.grad is not None};cg={n:p.grad for n,p in candidate.named_parameters() if p.grad is not None};assert set(grads)==set(cg)
  def diff(x,y):return {'max_abs':float((x-y).abs().max()),'close':bool(torch.allclose(x,y,atol=1e-6,rtol=1e-4))}
  report['output']=diff(reference,result);report['input_gradients']=[diff(x.grad,y.grad) for x,y in zip(a,b)];report['parameter_gradient_max_abs']=max(float((grads[n]-cg[n]).abs().max()) for n in grads);report['all_parameter_gradients_close']=all(torch.allclose(grads[n],cg[n],atol=1e-6,rtol=1e-4) for n in grads);report['all_gradients_finite']=all(bool(torch.isfinite(v).all()) for v in cg.values());save()
  for label,mod,inputs in [('eager',raw,a),('compiled',compiled,b),('compiled',compiled,b),('eager',raw,a)]:
   for _ in range(3):step(mod,inputs)
   torch.cuda.synchronize();t=time.perf_counter()
   for _ in range(20):step(mod,inputs)
   torch.cuda.synchronize();row={'mode':label,'steps':20,'ms_per_step':(time.perf_counter()-t)*1000/20};report['timings'].append(row);save();print('FUSION_TIMING',row,flush=True)
  report['complete']=True
 except BaseException as e:report['complete']=False;report['error']=repr(e);report['traceback']=traceback.format_exc();print('ERROR',repr(e),flush=True)
 save();print('FUSION_RESULT',json.dumps(report),flush=True)
if __name__=='__main__':main()

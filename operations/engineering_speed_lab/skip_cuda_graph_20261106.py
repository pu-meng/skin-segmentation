import os
os.environ.update(CUDA_VISIBLE_DEVICES='GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
import sys,json,time,copy,traceback
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation');OUT=ROOT/'operations/engineering_speed_lab/skip_cudagraph_20261106';OUT.mkdir(exist_ok=True)
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
import torch
from source.e1232_fourarm_screen_20261106 import entry as entry
from source.e123_joint_20261105.candidate import E3

def main():
 report={'scope':'complete dual E3 forward/backward CUDA graph; standalone FP32 batch8 C64 H64 W64 synthetic inputs; not deployed to live training','timings':[]}
 def save():(OUT/'result.json').write_text(json.dumps(report,indent=2))
 try:
  torch.set_num_threads(2);torch.cuda.set_device(0);torch.manual_seed(42)
  raw=E3().cuda().train();model=copy.deepcopy(raw);a=[torch.randn(8,64,64,64,device='cuda',requires_grad=True) for _ in range(2)];b=[x.detach().clone().requires_grad_() for x in a]
  def step(m,x):
   m.zero_grad(set_to_none=True)
   for z in x:z.grad=None
   o=m(*x);loss=sum(z.square().mean() for z in o);loss.backward();return o
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   for _ in range(3):step(model,b)
  torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
  model.zero_grad(set_to_none=True)
  for x in b:x.grad=None
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):out=step(model,b)
  graph.replay();torch.cuda.synchronize();reference=step(raw,a);torch.cuda.synchronize()
  report['output_max_abs']=max(float((x-y).abs().max()) for x,y in zip(reference,out));report['input_gradient_max_abs']=max(float((x.grad-y.grad).abs().max()) for x,y in zip(a,b));rg=dict(raw.named_parameters());cg=dict(model.named_parameters());report['parameter_gradient_max_abs']=max(float((p.grad-cg[n].grad).abs().max()) for n,p in rg.items() if p.grad is not None);report['gradients_finite']=all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None);save()
  for label,fn in [('eager',lambda:step(raw,a)),('graph',graph.replay),('graph',graph.replay),('eager',lambda:step(raw,a))]:
   for _ in range(3):fn()
   torch.cuda.synchronize();t=time.perf_counter()
   for _ in range(20):fn()
   torch.cuda.synchronize();row={'mode':label,'steps':20,'ms_per_step':(time.perf_counter()-t)*1000/20};report['timings'].append(row);save();print('GRAPH',row,flush=True)
  report['complete']=True
 except BaseException as e:report.update(complete=False,error=repr(e),traceback=traceback.format_exc());print(report['traceback'],flush=True)
 save();print('RESULT',json.dumps(report),flush=True)
if __name__=='__main__':main()

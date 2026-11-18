import os
os.environ.update(CUDA_VISIBLE_DEVICES='GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
import sys,json,time,copy,types,traceback
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation');OUT=ROOT/'operations/engineering_speed_lab/skip_concat_reuse_20261106';OUT.mkdir(exist_ok=True)
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
import torch
import torch.nn.functional as F
from source.e1232_fourarm_screen_20261106 import entry as entry
from source.e123_joint_20261105.candidate import E3

def reuse(self,z):
 a,b,c=z.split([24,24,16],1);t=self.n(self.fuse(torch.cat([self.lap(a),self.haar(b),self.fourier(c)],1)));a,b,c=t.split([24,24,16],1)
 a=a+self.rep(a);b=self.axis(b);c=c+F.interpolate(self.coarse(F.avg_pool2d(c,4,4)),size=c.shape[-2:],mode='bilinear',align_corners=False)
 joined=torch.cat([a,b,c],1);w=self.select(joined.mean((2,3),keepdim=True)).softmax(1)
 scale=torch.cat([w[:,0:1].expand(-1,24,-1,-1),w[:,1:2].expand(-1,24,-1,-1),w[:,2:3].expand(-1,16,-1,-1)],1)
 return self.out((3*scale)*joined)
def main():
 torch.set_num_threads(2);torch.cuda.set_device(0);torch.manual_seed(42);raw=E3().cuda().train();new=copy.deepcopy(raw)
 for side in [new.s,new.d]:side.gen.forward=types.MethodType(reuse,side.gen)
 a=[torch.randn(8,64,64,64,device='cuda',requires_grad=True) for _ in range(2)];b=[x.detach().clone().requires_grad_() for x in a]
 def step(m,x):
  m.zero_grad(set_to_none=True)
  for v in x:v.grad=None
  y=m(*x);sum(v.square().mean() for v in y).backward();return y
 x=step(raw,a);y=step(new,b);torch.cuda.synchronize();p=dict(raw.named_parameters());q=dict(new.named_parameters())
 r={'scope':'dual E3 full FP32 batch8 synthetic exact interface; reuse joined tensor for weighting; live training untouched','output_max_abs':max(float((v-w).abs().max()) for v,w in zip(x,y)),'input_gradient_max_abs':max(float((v.grad-w.grad).abs().max()) for v,w in zip(a,b)),'parameter_gradient_max_abs':max(float((v.grad-q[k].grad).abs().max()) for k,v in p.items() if v.grad is not None),'timings':[]}
 for label,m,inp in [('original',raw,a),('reuse',new,b),('reuse',new,b),('original',raw,a)]:
  for _ in range(3):step(m,inp)
  torch.cuda.synchronize();t=time.perf_counter()
  for _ in range(20):step(m,inp)
  torch.cuda.synchronize();r['timings'].append({'mode':label,'ms':(time.perf_counter()-t)*1000/20});(OUT/'result.json').write_text(json.dumps(r,indent=2));print(r['timings'][-1],flush=True)
 av=lambda mode:sum(v['ms'] for v in r['timings'] if v['mode']==mode)/2
 r['time_reduction_percent']=100*(1-av('reuse')/av('original'));r['complete']=True;(OUT/'result.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
if __name__=='__main__':main()

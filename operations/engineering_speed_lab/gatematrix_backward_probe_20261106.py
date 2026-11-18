import os
os.environ.update(CUDA_VISIBLE_DEVICES='GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
import sys,json,collections,signal,time
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation');OUT=ROOT/'operations/engineering_speed_lab/gatematrix_backward_20261106';OUT.mkdir(exist_ok=True)
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
import torch
from torch.profiler import profile,ProfilerActivity,record_function
from source.e1232_fourarm_screen_20261106 import entry as m
from source.e1232_matrix_20261106.candidate import build_candidate
from ham_fast_pipeline import PlannedDataset

def main():
 torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.16)
 cfg=json.loads(Path('/media/8T/PuMengYu/experiments/skin_segmentation/E1232_GateMatrix_FRESH300_20261106/protocol.json').read_text())['config']
 ds=PlannedDataset(Path(cfg['data']['split_file']).parent,'train');rows=[ds[i] for i in range(8)]
 x=torch.stack([z[0] for z in rows]).cuda();y=torch.stack([z[1] for z in rows]).cuda();api=m.runner.baseline_api();api.seed_all(42);model=build_candidate('BT',False).cuda().train();crit=api.BceDiceLoss().cuda()
 def step():
  model.zero_grad(set_to_none=True)
  with record_function('FORWARD'):p=model(x);loss=crit(p,y)
  with record_function('BACKWARD'):loss.backward()
 for _ in range(2):step()
 torch.cuda.synchronize();stacks=collections.defaultdict(list);handles=[]
 def pre(name):
  def f(mod,args):
   c=record_function('MODULE::'+name);c.__enter__();stacks[id(mod)].append(c)
  return f
 def post(mod,args,out):stacks[id(mod)].pop().__exit__(None,None,None)
 for name,mod in model.named_modules():handles.extend([mod.register_forward_pre_hook(pre(name)),mod.register_forward_hook(post)])
 with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=False,with_stack=False) as prof:
  for _ in range(3):step();prof.step()
 torch.cuda.synchronize()
 for h in handles:h.remove()
 prof.export_chrome_trace(str(OUT/'trace.json'))
 (OUT/'profile.json').write_text(json.dumps({'steps':3,'batch':8,'precision':'FP32','scope':'isolated BT random weights real train inputs; shared GPU; no formal changes','pid':os.getpid()}))
 print('PROFILE_READY',str(OUT),flush=True)
if __name__=='__main__':main()

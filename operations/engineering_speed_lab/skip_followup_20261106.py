import os
os.environ.update(CUDA_VISIBLE_DEVICES='GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
import sys,time,json,copy,statistics,collections,gc
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation');sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
import torch
from source.e1232_fourarm_screen_20261106 import entry as entry
from source.e123_joint_20261105.candidate import E3
OUT=ROOT/'operations/engineering_speed_lab/skip_cost_parts_20261106'
torch.cuda.set_device(0);torch.manual_seed(42);mod=E3().cuda().train();s=torch.randn(8,64,64,64,device='cuda',requires_grad=True);d=torch.randn_like(s,requires_grad=True);results=[]
def step():
 mod.zero_grad(set_to_none=True);s.grad=d.grad=None;a,b=mod(s,d);loss=a.square().mean()+b.square().mean();loss.backward()
for threads in [2,1,4,4,1,2]:
 torch.set_num_threads(threads)
 for _ in range(3):step()
 torch.cuda.synchronize();t=time.perf_counter()
 for _ in range(8):step()
 torch.cuda.synchronize();row={'cpu_threads':threads,'ms_per_step':(time.perf_counter()-t)/8*1000};results.append(row);print('THREAD',row,flush=True)
torch.set_num_threads(2)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
 for _ in range(3):step();prof.step()
torch.cuda.synchronize();prof.export_chrome_trace(str(OUT/'full_skip_trace.json'))
events=json.loads((OUT/'full_skip_trace.json').read_text())['traceEvents'];kernels=[e for e in events if e.get('ph')=='X' and e.get('cat') in ['kernel','gpu_memset','gpu_memcpy']];runtimes=[e for e in events if e.get('ph')=='X' and e.get('cat')=='cuda_runtime'];cpu=collections.defaultdict(lambda:[0,0])
for e in events:
 if e.get('ph')=='X' and e.get('cat')=='cpu_op':cpu[e['name']][0]+=e['dur'];cpu[e['name']][1]+=1
report={'threads':results,'gpu_kernel_ms_per_step':sum(e['dur'] for e in kernels)/3000,'gpu_kernel_count_per_step':len(kernels)/3,'cuda_runtime_ms_per_step':sum(e['dur'] for e in runtimes)/3000,'cpu_inclusive_top10':[{'op':k,'ms_per_step':v[0]/3000,'calls_per_step':v[1]/3} for k,v in sorted(cpu.items(),key=lambda z:-z[1][0])[:10]],'note':'CPU inclusive values overlap; synthetic exact interface; sharedGPU'}
(OUT/'full_skip_followup.json').write_text(json.dumps(report,indent=2));print('FOLLOWUP',json.dumps(report),flush=True)

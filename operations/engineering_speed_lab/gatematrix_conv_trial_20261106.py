import os
os.environ.update(CUDA_VISIBLE_DEVICES='GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
import sys,time,json,copy
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation');sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
import torch
from source.e1232_fourarm_screen_20261106 import entry as m
from source.e1232_matrix_20261106.candidate import build_candidate
OUT=ROOT/'operations/engineering_speed_lab/gatematrix_backward_20261106'
torch.set_num_threads(2);torch.manual_seed(42);torch.cuda.set_device(0)
model=build_candidate('BT',False);block=copy.deepcopy(model.decoder.stages[1]);del model
state=copy.deepcopy(block.state_dict());x=torch.randn(8,256,32,32);target=torch.randn(8,128,32,32);results=[];reference=None
for fmt in ['nchw','channels_last','channels_last','nchw']:
 b=copy.deepcopy(block).cuda();b.load_state_dict(state);b.train();xx=x.cuda().requires_grad_();yy=target.cuda()
 if fmt=='channels_last':b=b.to(memory_format=torch.channels_last);xx=xx.detach().to(memory_format=torch.channels_last).requires_grad_()
 def step():
  b.zero_grad(set_to_none=True);xx.grad=None;p=b(xx);loss=(p-yy).square().mean();loss.backward();return p
 for _ in range(2):step()
 torch.cuda.synchronize();t=time.perf_counter()
 for _ in range(12):p=step()
 torch.cuda.synchronize();seconds=time.perf_counter()-t
 vals={'output':p.detach().cpu(),'input_grad':xx.grad.detach().cpu(),**{n:v.grad.detach().cpu() for n,v in b.named_parameters()}}
 if reference is None:reference=vals
 errors={n:float((v-reference[n]).abs().max()) for n,v in vals.items()}
 row=dict(format=fmt,steps=12,ms_per_step=seconds/12*1000,output_max_diff=errors['output'],input_grad_max_diff=errors['input_grad'],max_parameter_grad_diff=max(v for k,v in errors.items() if k not in ['output','input_grad']),finite=all(bool(torch.isfinite(v).all()) for v in vals.values()))
 results.append(row);print(row,flush=True);del b,xx,yy,p,vals;torch.cuda.empty_cache()
(OUT/'channels_last_trial.json').write_text(json.dumps({'scope':'D1 DoubleConv256to128 batch8 32x32 forward+backward only; ABBA sharedGPU, not fullmodel speed','results':results},indent=2))

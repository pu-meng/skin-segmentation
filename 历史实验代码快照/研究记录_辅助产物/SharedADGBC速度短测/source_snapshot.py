from pathlib import Path
import sys,json,time,statistics
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison')]
import torch
from Models.SharedADGBC.model import TinyViMSharedADGBC
from pretrained_unet2d import PretrainedUNet2D
from ham_fast_runner import baseline_api

torch.set_num_threads(4);torch.manual_seed(42)
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
torch.cuda.set_device(0)
base=PretrainedUNet2D('tinyvim_s',pretrained=False,new_layers_seed=42)
shared=TinyViMSharedADGBC(pretrained=False)
missing,unexpected=shared.load_state_dict(base.state_dict(),strict=False)
assert not unexpected and all(k.startswith('shared_adgbc.') for k in missing)
models={'base':base.cuda(),'shared':shared.cuda()}
optimizers={k:torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01) for k,m in models.items()}
criterion=baseline_api().BceDiceLoss().cuda()
x=torch.rand(8,3,256,256,device='cuda');y=(torch.rand(8,1,256,256,device='cuda')>.5).float()

def step(key):
 m=models[key];m.train();optimizers[key].zero_grad(set_to_none=True)
 loss=criterion(m(x),y);loss.backward();optimizers[key].step()

def timed(fn):
 torch.cuda.synchronize();t=time.perf_counter();fn();torch.cuda.synchronize();return (time.perf_counter()-t)*1000

for key in models:
 for _ in range(2):step(key)
times={k:[] for k in models}
for rep in range(4):
 for key in (('base','shared') if rep%2==0 else ('shared','base')):
  times[key].append(timed(lambda:step(key)))
 print('PAIR',rep,{k:round(v[-1],2) for k,v in times.items()},flush=True)
inference={}
for batch in (1,8):
 inference[str(batch)]={}
 with torch.inference_mode():
  for key,m in models.items():
   m.eval();m(x[:batch]);torch.cuda.synchronize()
  for key,m in models.items():
   vals=[timed(lambda:m(x[:batch])) for _ in range(4)]
   inference[str(batch)][key]={'ms':vals,'mean_ms':statistics.mean(vals)}
profiles={}
for key in models:
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=True) as prof:
  step(key);torch.cuda.synchronize()
 events=prof.key_averages(group_by_input_shape=True)
 rows=[]
 for e in events:
  if e.self_cuda_time_total>0:
   rows.append({'op':e.key,'shapes':e.input_shapes,'calls':e.count,'self_cuda_ms':e.self_cuda_time_total/1000,'cuda_ms':e.cuda_time_total/1000})
 rows.sort(key=lambda r:r['self_cuda_ms'],reverse=True)
 profiles[key]={'summed_self_cuda_ms':sum(r['self_cuda_ms'] for r in rows),'top20':rows[:20]}
 print('PROFILE',key,profiles[key]['summed_self_cuda_ms'],rows[:5],flush=True)
result={'scope':'synthetic same initial common weights, batch8, FP32, deterministic cuDNN, AdamW; GPU0 concurrently busy with other tasks; not isolated throughput','training_step':{k:{'ms':v,'mean_ms':statistics.mean(v)} for k,v in times.items()},'inference':inference,'profiles':profiles,'existing_experiments_modified':False,'test_data_accessed':False}
out=ROOT/'log/研究记录_辅助产物/SharedADGBC速度短测/profile.json';out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('RESULT',json.dumps({k:v for k,v in result.items() if k!='profiles'},ensure_ascii=False),flush=True)

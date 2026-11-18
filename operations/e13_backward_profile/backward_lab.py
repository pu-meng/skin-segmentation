import os,sys,json,time,subprocess,traceback,collections
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation');OUT=ROOT/'operations/e13_backward_profile'
os.environ.update(CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',MKL_THREADING_LAYER='GNU',OPENBLAS_NUM_THREADS='1')
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
def save(name,d):
 p=OUT/name;p.write_text(json.dumps(d,indent=2,ensure_ascii=False))
def gpu():return subprocess.getoutput('nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader')
def main():
 import torch,random,numpy as np,inspect
 from pretrained_unet2d import prepare_imports
 prepare_imports('mambavision_t');prepare_imports('tinyvim_s')
 from source.e1_e3_combo.candidate import build_candidate
 import ham_fast_runner as runner
 from ham_fast_pipeline import PlannedDataset
 from project_paths import current_protocol
 from torch.profiler import profile,ProfilerActivity,record_function
 torch.set_num_threads(4);api=runner.baseline_api();api.seed_all(42)
 model=build_candidate(pretrained=True,arm='E1_E3')
 checkpoints=[]
 for n,m in model.named_modules():
  attrs={k:v for k,v in vars(m).items() if 'checkpoint' in k.lower() and isinstance(v,(str,bool,int,float,type(None)))}
  if attrs:checkpoints.append(dict(module=n,type=type(m).__name__,values=attrs))
 import selective_scan_cuda
 audit=dict(pid=os.getpid(),checkpoints=checkpoints,scan_extension=selective_scan_cuda.__file__,torch=torch.__version__,gpu_before=gpu(),cudnn=dict(benchmark=torch.backends.cudnn.benchmark,deterministic=torch.backends.cudnn.deterministic,allow_tf32=torch.backends.cudnn.allow_tf32),cuda_initialized_before_transfer=torch.cuda.is_initialized())
 save('audit.json',audit)
 cfg=current_protocol(ROOT/'source/config/protocol_v5_dedup_seed42.json');ds=PlannedDataset(Path(cfg['data']['split_file']).parent,'train')
 plans=json.loads((ROOT/'operations/e13_slow_isolation_20261105/epoch_plan.json').read_text())['plans'][:24]
 batches=[]
 for i in range(3):
  items=[ds[tuple(p)] for p in plans[i*8:(i+1)*8]]
  batches.append((torch.stack([t[0] for t in items]),torch.stack([t[1] for t in items])))
 torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.20,0)
 model=model.cuda().train();crit=api.BceDiceLoss().cuda();batches=[(x.cuda(),y.cuda()) for x,y in batches]
 initial={n:t.detach().cpu().clone() for n,t in model.state_dict().items()}
 def step(b):
  model.zero_grad(set_to_none=True);x,y=b;p=model(x);loss=crit(p,y);loss.backward();return p,loss
 for i in range(2):step(batches[i]);torch.cuda.synchronize()
 handles=[];stacks=collections.defaultdict(list)
 def pre(name):
  def f(m,args):
   ctx=record_function('MODULE::'+name);ctx.__enter__();stacks[id(m)].append(ctx)
  return f
 def post(m,args,out):stacks[id(m)].pop().__exit__(None,None,None)
 for n,m in model.named_modules():
  handles.extend([m.register_forward_pre_hook(pre(n)),m.register_forward_hook(post)])
 torch.cuda.reset_peak_memory_stats()
 with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True,with_stack=False,profile_memory=False) as prof:
  for i,b in enumerate(batches):
   with record_function('PROFILE_STEP_'+str(i)):
    model.zero_grad(set_to_none=True)
    with record_function('FORWARD'):p=model(b[0]);loss=crit(p,b[1])
    with record_function('BACKWARD'):loss.backward()
   prof.step()
 torch.cuda.synchronize()
 for h in handles:h.remove()
 prof.export_chrome_trace(str(OUT/'trace.json'))
 events=list(prof.events());seqmod={}
 def parents(e):
  while e is not None:
   yield e;e=e.cpu_parent
 for e in events:
  if e.sequence_nr>=0:
   module=next((a.name[8:] for a in parents(e) if a.name.startswith('MODULE::')),None)
   if module is not None:seqmod[e.sequence_nr]=module
 ops=collections.defaultdict(lambda:[0.,0]);mods=collections.defaultdict(float);cats=collections.defaultdict(float)
 for e in events:
  chain=list(parents(e));back=any(a.name=='BACKWARD' for a in chain)
  if not back:continue
  us=float(e.self_cuda_time_total)
  if not us:continue
  ops[e.name][0]+=us;ops[e.name][1]+=1
  module=next((seqmod[a.sequence_nr] for a in chain if a.sequence_nr in seqmod),'unattributed')
  mods[module]+=us
  n=e.name.lower()
  cat='conv_backward' if 'convolution_backward' in n or 'cudnn_convolution' in n else 'selective_scan_backward' if 'selectivescan' in n else 'batchnorm_backward' if 'batch_norm' in n else 'matrix_ffn' if any(s in n for s in ('mm','einsum')) else 'copy_reduction' if any(s in n for s in ('copy','sum','mean','clone','contiguous')) else 'other'
  cats[cat]+=us
 total=sum(v[0] for v in ops.values())
 result=dict(pid=os.getpid(),steps=3,batch=8,precision='FP32',shared_gpu=True,backward_self_cuda_total_ms=total/1000,top15=[dict(op=k,self_cuda_ms=v[0]/1000,ms_per_step=v[0]/3000,calls=v[1],percent=v[0]/total*100) for k,v in sorted(ops.items(),key=lambda x:-x[1][0])[:15]],module_top20=[dict(module=k,self_cuda_ms=v/1000,ms_per_step=v/3000) for k,v in sorted(mods.items(),key=lambda x:-x[1])[:20]],categories_ms={k:v/1000 for k,v in cats.items()},peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,gpu_after_profile=gpu(),module_attribution='forward op sequence_nr to backward ancestor; each self CUDA event counted once; leaf module attribution, not inclusive module sum')
 save('profile.json',result);print('PROFILE',json.dumps(result),flush=True)
 # Minimal engineering trial: permit cuDNN algorithm search/workspace, retain FP32, no scan/math/gradient changes.
 timings=[];numerical={};reference=None
 for mode in ('baseline','workspace','workspace','baseline'):
  model.load_state_dict(initial);torch.backends.cudnn.benchmark=mode=='workspace';torch.backends.cudnn.deterministic=mode=='baseline'
  step(batches[0]);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
  spans=[]
  for b in batches:
   model.zero_grad(set_to_none=True);a,bw,c=[torch.cuda.Event(enable_timing=True) for _ in range(3)]
   a.record();p=model(b[0]);loss=crit(p,b[1]);bw.record();loss.backward();c.record();spans.append((a,bw,c))
  torch.cuda.synchronize()
  timings.append(dict(mode=mode,forward_ms=[a.elapsed_time(b) for a,b,c in spans],backward_ms=[b.elapsed_time(c) for a,b,c in spans],peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))
  payload={n:t.grad.detach().cpu().clone() for n,t in model.named_parameters() if t.grad is not None};pred=p.detach().cpu()
  if reference is None:reference=(pred,payload)
  else:
   diffs=[(payload[n]-reference[1][n]).abs().max().item() for n in payload]
   numerical[str(len(timings))]=dict(mode=mode,prediction_max_abs=(pred-reference[0]).abs().max().item(),gradient_max_abs=max(diffs),all_gradients_finite=all(torch.isfinite(v).all().item() for v in payload.values()))
  save('trial.json',dict(timings=timings,numerical=numerical,note='3 measured steps per arm after one warmup; ABBA; algorithm nondeterminism relaxed only in workspace arm; not a guaranteed numerically identical replacement'))
 print('TRIAL',json.dumps(timings),flush=True)
 save('done.json',dict(pid=os.getpid(),complete=True,time=time.time(),gpu=gpu()))
if __name__=='__main__':
 OUT.mkdir(exist_ok=True)
 try:main()
 except BaseException as e:save('failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise

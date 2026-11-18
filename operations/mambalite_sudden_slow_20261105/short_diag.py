import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['OMP_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
import sys,time,json,hashlib,subprocess,itertools
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
SRC=ROOT/'source/mambalite_paper_candidate_20261105'
OUT=ROOT/'operations/mambalite_sudden_slow_20261105'
OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(SRC))
import torch
from experiment import build_model,api,PlannedDataset,SPLIT,SeedPlanSampler

def save(name,obj):
 (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def snap():
 return {'time':time.time(),'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm','--format=csv,noheader'],text=True),'apps':subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True),'status':json.loads((ROOT/'log/mambalite_fig2b_20261105/seed42/status.json').read_text())}
def proc():
 p=Path('/proc/2534485');s=(p/'stat').read_text().split();return {'cpu_ticks':int(s[13])+int(s[14]),'minor_faults':int(s[9]),'major_faults':int(s[11]),'schedstat':(p/'schedstat').read_text(),'io':(p/'io').read_text()}
def timed(fn):
 torch.cuda.synchronize();t=time.monotonic();v=fn();torch.cuda.synchronize();return v,time.monotonic()-t
c=json.loads((SRC/'protocol.json').read_text());before={p:sha(p) for p in c['source_hashes']}
result={'pid':os.getpid(),'torch':torch.__version__,'before':snap(),'proc_before':proc(),'source_before':before,'boundary':'Three real augmented batch8 FP32 steps; frozen epoch48 checkpoint, original AdamW state; no official writes. Phases deliberately synchronized; not full epoch throughput. Gate ABBA uses identical gradient tensors with no mutation.'}
save('report.json',result)
free,total=torch.cuda.mem_get_info()
assert free>5*1024**3,('insufficient headroom',free)
torch.cuda.set_per_process_memory_fraction(.18)
api.seed_all(42);torch.set_num_threads(4)
checkpoint=ROOT/'Models/mambalite_fig2b_20261105/seed42/latest.pth'
# Hash the bytes loaded, avoiding races with the next official checkpoint replacement.
import io
raw=checkpoint.read_bytes();result['checkpoint_sha256']=hashlib.sha256(raw).hexdigest()
state=torch.load(io.BytesIO(raw),map_location='cpu');del raw
result['checkpoint_epoch']=state['epoch']
m=build_model().cuda();m.load_state_dict(state['model'],strict=True);m.train()
opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8);opt.load_state_dict(state['optimizer']);del state
criterion=api.BceDiceLoss().cuda()
ds=PlannedDataset(SPLIT.parent,'train');plan=list(itertools.islice(SeedPlanSampler(ds,42),24));save('plan.json',plan)
rows=[]
for i in range(3):
 t=time.monotonic();items=[ds[p] for p in plan[i*8:(i+1)*8]];x=torch.stack([v[0] for v in items]);y=torch.stack([v[1] for v in items]);load=time.monotonic()-t
 ids=[v[2] for v in items];input_sha=hashlib.sha256(x.numpy().tobytes()+y.numpy().tobytes()).hexdigest()
 _,h2d=timed(lambda:None)
 t=time.monotonic();x=x.cuda();y=y.cuda();torch.cuda.synchronize();h2d=time.monotonic()-t
 opt.zero_grad(set_to_none=True)
 def fw():
  p=m(x);return criterion(p,y)
 loss,forward=timed(fw)
 loss_ok,losscheck=timed(lambda:bool(torch.isfinite(loss)))
 _,backward=timed(loss.backward)
 grads=[p.grad for p in m.parameters() if p.grad is not None]
 def original():return not any(not bool(torch.isfinite(g).all()) for g in grads)
 def aggregate():return bool(torch.stack([torch.isfinite(g).all() for g in grads]).all())
 checks=[]
 for mode,fn in [('original',original),('aggregate',aggregate),('aggregate',aggregate),('original',original)]:
  ok,seconds=timed(fn);checks.append({'mode':mode,'finite':ok,'seconds':seconds})
 assert loss_ok and all(c['finite'] for c in checks)
 # Check equal failure behavior using a separate cloned gradient (never mutate model gradients).
 if i==0:
  probe=grads[0].detach().clone();probe.reshape(-1)[0]=float('nan')
  result['nonfinite_probe']={'original':not any(not bool(torch.isfinite(g).all()) for g in [probe]),'aggregate':bool(torch.stack([torch.isfinite(g).all() for g in [probe]]).all())}
  del probe
 _,update=timed(opt.step)
 rows.append({'step':i+1,'ids':ids,'input_sha256':input_sha,'loss':float(loss.detach()),'data_cpu_seconds':load,'h2d_seconds':h2d,'forward_loss_seconds':forward,'loss_check_seconds':losscheck,'backward_seconds':backward,'gradient_tensors':len(grads),'checks':checks,'optimizer_seconds':update})
 result['steps']=rows;save('report.json',result);print(json.dumps(rows[-1]),flush=True)
result.update(after=snap(),proc_after=proc(),source_after={p:sha(p) for p in before},peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),complete=True)
assert result['source_before']==result['source_after']
save('report.json',result)
print('DONE',OUT/'report.json',flush=True)

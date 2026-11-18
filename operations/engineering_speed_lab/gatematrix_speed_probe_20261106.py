import os
os.environ['CUDA_VISIBLE_DEVICES']='GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b'
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,time,json,signal
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation');sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
import torch
from source.e1232_fourarm_screen_20261106 import entry as m
from source.e1232_matrix_20261106.candidate import build_candidate
OUT=ROOT/'operations/engineering_speed_lab/gatematrix_short_probe_20261106'
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 result={'scope':'isolated BT random initialization, real training batches; synchronized phase timings under concurrent load; not formal epoch or accuracy test','train':[],'val':[]};loaders={}
 def save(): (OUT/'result.json').write_text(json.dumps(result,indent=2))
 def expired(*args):raise TimeoutError('short probe 150s budget')
 signal.signal(signal.SIGALRM,expired);signal.alarm(150)
 try:
  torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.16)
  config=json.loads(Path('/media/8T/PuMengYu/experiments/skin_segmentation/E1232_GateMatrix_FRESH300_20261106/protocol.json').read_text())['config']
  api=m.runner.baseline_api();api.seed_all(42)
  model=build_candidate('BT',False).cuda();criterion=api.BceDiceLoss().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
  loaders,g=m.make_fast_loaders(config,api,workers=2,validation_batch=8);it=iter(loaders['train']);model.train()
  for k in range(6):
   torch.cuda.synchronize();t=time.perf_counter();x,y,ids,_=next(it);t1=time.perf_counter();x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);torch.cuda.synchronize();t2=time.perf_counter()
   opt.zero_grad(set_to_none=True);pred=model(x);loss=criterion(pred,y);torch.cuda.synchronize();t3=time.perf_counter();loss.backward();torch.cuda.synchronize();t4=time.perf_counter();opt.step();torch.cuda.synchronize();t5=time.perf_counter()
   row=dict(step=k,data_wait=t1-t,h2d=t2-t1,forward_loss=t3-t2,backward=t4-t3,optimizer=t5-t4,total=t5-t,warmup=k<2);result['train'].append(row);save();print('TRAIN_TIMING',json.dumps(row),flush=True)
  model.eval();ds=loaders['val'].dataset;batch=[ds[i] for i in range(16)];xs=torch.stack([z[0] for z in batch]).cuda();ys=torch.stack([z[1] for z in batch]).cuda();reference=None
  with torch.no_grad():
   for bs in [1,2,8,8,2,1]:
    model(xs[:bs]);torch.cuda.synchronize();t=time.perf_counter();preds=[]
    for j in range(0,16,bs):
     p=model(xs[j:j+bs]);v=criterion(p,ys[j:j+bs]);preds.append(p);v.item();p[:,0].cpu().numpy()
    torch.cuda.synchronize();seconds=time.perf_counter()-t;v=torch.cat(preds).cpu()
    if reference is None:reference=v
    row=dict(batch=bs,images=16,seconds=seconds,max_output_diff=float((v-reference).abs().max()));result['val'].append(row);save();print('VAL_TIMING',json.dumps(row),flush=True)
  result['peak_reserved_mib']=torch.cuda.max_memory_reserved()/2**20;result['complete']=True
 except BaseException as e:result['error']=repr(e);result['complete']=False;print('PROBE_ERROR',repr(e),flush=True)
 finally:
  signal.alarm(0)
  for loader in loaders.values():
   base=getattr(loader,'loader',loader);it=getattr(base,'_iterator',None)
   if it is not None:it._shutdown_workers()
  save()
if __name__=='__main__':main()

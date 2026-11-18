def main():
 import os,sys,json,time,subprocess,csv,hashlib,fcntl
 from pathlib import Path
 ROOT=Path('/home/PuMengYu/experiments/skin_segmentation');HERE=ROOT/'source/e123_joint_20261105';RUN=ROOT/'log/TinyViM_E123.1_联合新版_HAM_20261105';OUT=Path(__file__).parent
 UUID='GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b'
 os.environ.update(CUDA_VISIBLE_DEVICES=UUID,EXPERIMENT_PHYSICAL_GPU='1',MKL_THREADING_LAYER='GNU',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
 sys.path[:0]=[str(HERE),str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
 with (OUT/'profile.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  # Keep first two formal full-epoch measurements free of this profiler's GPU load.
  deadline=time.monotonic()+600
  while True:
   state=json.loads((RUN/'controller_status.json').read_text());assert state['status']!='failed',state
   p=RUN/'run/history.csv';rows=list(csv.DictReader(p.open())) if p.exists() else []
   if len(rows)>=2:break
   assert time.monotonic()<deadline,'deadline waiting two complete epochs'
   time.sleep(5)
  if float(rows[-1]['seconds'])<=70:
   (OUT/'report.json').write_text(json.dumps(dict(skipped=True,reason='latest full formal epoch <=70s; do not optimize fast group',history=rows),indent=2));raise SystemExit(0)
  free=int(subprocess.check_output(['nvidia-smi','-i',UUID,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>=6500,free
  import torch
  from pretrained_unet2d import prepare_imports
  prepare_imports('mambavision_t');prepare_imports('tinyvim_s')
  from source.e123_joint_20261105.candidate import build_candidate
  import ham_fast_runner as runner
  from ham_fast_pipeline import make_fast_loaders
  api=runner.baseline_api();api.seed_all(42);torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.25)
  config=runner.validate_protocol(runner.PROTOCOL,'tinyvim_s',check_cache=False)
  model=build_candidate(pretrained=True).cuda().train();loaders,generator=make_fast_loaders(config,api,workers=2,validation_batch=8)
  opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8);crit=api.BceDiceLoss().cuda()
  started=time.time();before=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
  try:
   it=iter(loaders['train']);waits=[];losses=[]
   for i in range(3):
    t=time.perf_counter();x,y,ids,_=next(it);waits.append(time.perf_counter()-t);opt.zero_grad(set_to_none=True);loss=crit(model(x.cuda()),y.cuda());loss.backward();opt.step();losses.append(float(loss.detach()))
   handles=[];contexts={}
   for name,mod in [('E1',model.decoder.stages[3]),('E2',model.decoder.e2),('E3',model.decoder.e3)]:
    def pre(m,a,name=name):contexts[name]=torch.profiler.record_function(name+'_FORWARD');contexts[name].__enter__()
    def post(m,a,o,name=name):contexts.pop(name).__exit__(None,None,None)
    handles.extend([mod.register_forward_pre_hook(pre),mod.register_forward_hook(post)])
   with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=True,with_stack=False) as p:
    for i in range(3):
     with torch.profiler.record_function('DATA_WAIT'):
      t=time.perf_counter();x,y,ids,_=next(it);waits.append(time.perf_counter()-t)
     opt.zero_grad(set_to_none=True)
     with torch.profiler.record_function('FORWARD_LOSS'):loss=crit(model(x.cuda()),y.cuda())
     with torch.profiler.record_function('BACKWARD'):loss.backward()
     with torch.profiler.record_function('OPTIMIZER'):opt.step()
     losses.append(float(loss.detach()));p.step()
   torch.cuda.synchronize();p.export_chrome_trace(str(OUT/'trace.json'))
   events=p.key_averages();table=events.table(sort_by='self_cuda_time_total',row_limit=35);(OUT/'operator_table.txt').write_text(table)
   groups=[dict(name=e.key,cpu_total_us=e.cpu_time_total,cuda_total_us=e.cuda_time_total,self_cuda_us=e.self_cuda_time_total,count=e.count) for e in events if e.key in ['DATA_WAIT','FORWARD_LOSS','BACKWARD','OPTIMIZER','E1_FORWARD','E2_FORWARD','E3_FORWARD']]
   report=dict(complete=True,pid=os.getpid(),gpu_uuid=UUID,shared_gpu=True,formal_model_unchanged=True,source_sha256=hashlib.sha256((HERE/'candidate.py').read_bytes()).hexdigest(),batch_size=8,input_size=256,precision='FP32 unchanged',warm_steps=3,profile_steps=3,losses=losses,data_wait_seconds=waits,groups=groups,started_at=started,ended_at=time.time(),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,compute_before=before,compute_after=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True),formal_epochs_before=rows,scope='isolated initial weights real HAM batches, not epoch or performance evaluation; CUDA times include shared scheduling; profiler overhead; no optimized variant promoted')
   (OUT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True);print(table,flush=True)
  finally:
   for loader in loaders.values():
    if hasattr(loader,'close'):loader.close()

if __name__=='__main__':main()

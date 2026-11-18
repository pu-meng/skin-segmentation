import os,sys,time,json,subprocess,signal,fcntl,hashlib,traceback
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
OUT=Path(__file__).resolve().parent
PY='/home/PuMengYu/anaconda3/envs/medseg/bin/python'
def write(name,value):
 p=OUT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2,ensure_ascii=False));t.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def snapshot():
 d={}
 for p in sorted((ROOT/'log').glob('**/history.csv')):
  d[str(p)]=dict(bytes=p.stat().st_size,mtime=p.stat().st_mtime,last=p.read_text().splitlines()[-1:])
 return dict(time=time.time(),gpu=subprocess.getoutput('nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader'),compute=subprocess.getoutput('nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader'),processes=subprocess.getoutput('ps -u PuMengYu -o pid,ppid,etime,args'),histories=d)
def controller():
 with (OUT/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  before=snapshot();write('protected_before.json',before)
  proto=json.loads((ROOT/'log/TinyViM_E1E3_HAM_20261105/protocol.json').read_text())
  sources={str(ROOT/p):sha(ROOT/p) for p in proto['source_sha256']}
  write('source_before.json',sources)
  state=dict(status='starting',controller_pid=os.getpid(),budget_seconds=900,started=time.time(),root=str(OUT),gpu=0)
  write('controller_status.json',state)
  child=subprocess.Popen([PY,'-B','-u',str(Path(__file__).resolve()),'--child'],cwd=ROOT,start_new_session=True)
  state.update(status='running',child_pid=child.pid,child_pgid=child.pid);write('controller_status.json',state)
  deadline=time.monotonic()+900
  try:
   while child.poll() is None and time.monotonic()<deadline:time.sleep(2)
   if child.poll() is None:
    state['timeout']=True;os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   state.update(status='complete' if child.returncode==0 else 'failed',exitcode=child.returncode,finished=time.time())
  finally:
   if child.poll() is None:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=10)
   after=snapshot();write('protected_after.json',after)
   source_after={p:sha(p) for p in sources};write('source_after.json',source_after)
   state['source_unchanged']=sources==source_after
   state['history_advanced']={p:after['histories'].get(p,{}).get('bytes',0)>v['bytes'] for p,v in before['histories'].items()}
   write('controller_status.json',state)
   print('CONTROLLER FINAL',json.dumps(state),flush=True)
def child_main():
 global torch,np,random
 os.environ.update(CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',TORCHINDUCTOR_CACHE_DIR=str(OUT/'inductor_cache'),TRITON_CACHE_DIR=str(OUT/'triton_cache'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',MKL_THREADING_LAYER='GNU',OPENBLAS_NUM_THREADS='1')
 sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
 import random,numpy as np,torch,copy,gc
 from torch.utils.data import DataLoader
 from pretrained_unet2d import prepare_imports
 prepare_imports('mambavision_t');prepare_imports('tinyvim_s')
 from source.e1_e3_combo.candidate import build_candidate
 import ham_fast_runner as runner
 from ham_fast_pipeline import PlannedDataset,AugmentationPlanSampler,make_fast_loaders,worker_init,evaluate_fast
 from project_paths import current_protocol
 from speed_contract import valid_budget
 torch.set_num_threads(4)
 torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(0.20,0)
 api=runner.baseline_api();api.seed_all(42)
 cfg=current_protocol(ROOT/'source/config/protocol_v5_dedup_seed42.json')
 assert sha(cfg['data']['split_file'])==cfg['data']['split_sha256']
 model=build_candidate(pretrained=True,arm='E1_E3')
 initial={n:t.detach().cpu().clone() for n,t in model.state_dict().items()}
 torch.save(initial,OUT/'initial_state.pth')
 write('weight_loading_audit.json',model.encoder.loading_audit)
 refloaders,generator=make_fast_loaders(cfg,api,workers=2,validation_batch=8)
 torch.empty((),dtype=torch.int64).random_(generator=generator)
 plans=list(refloaders['train'].loader.sampler)
 for l in refloaders.values():
  if hasattr(l,'close'):l.close()
 del refloaders
 write('epoch_plan.json',dict(plans=plans,split_sha256=cfg['data']['split_sha256']))
 ds=PlannedDataset(Path(cfg['data']['split_file']).parent,'train');vd=PlannedDataset(Path(cfg['data']['split_file']).parent,'val')
 assert valid_budget(len(ds),8,(len(plans)+7)//8,len(vd))
 rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state())
 def reset():
  model.load_state_dict(initial,strict=True);random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2]);torch.cuda.set_rng_state(rng[3]);model.train()
 def loader(dataset,workers,batch,sampler=None):
  return DataLoader(dataset,batch_size=batch,sampler=sampler,num_workers=workers,pin_memory=True,drop_last=False,persistent_workers=True,prefetch_factor=2,multiprocessing_context='spawn',worker_init_fn=worker_init,generator=torch.Generator().manual_seed(42001),timeout=60)
 def close(l):
  if getattr(l,'_iterator',None) is not None:l._iterator._shutdown_workers()
 def optimizer():return torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8)
 def compare(a,b):
  details={};passed=True
  assert a.keys()==b.keys()
  for n,x in a.items():
   y=b[n];assert x.shape==y.shape
   if x.is_floating_point():
    diff=(x.double()-y.double());ma=float(diff.abs().max());rel=float(diff.norm()/(x.double().norm()+1e-12));ok=torch.allclose(x,y,atol=1e-6,rtol=1e-4)
   else:ma=float((x-y).abs().max());rel=0.;ok=torch.equal(x,y)
   details[n]=dict(max_abs=ma,relative_l2=rel,close=ok);passed=passed and ok
  return dict(passed=passed,tensors=len(details),max_abs=max(v['max_abs'] for v in details.values()),max_relative_l2=max(v['relative_l2'] for v in details.values()),per_tensor=details)
 result=dict(model='E1_E3 real build_candidate',pid=os.getpid(),physical_gpu=0,precision='FP32 unchanged',batch=8,input=[256,256],steps=873,train_images=6983,val_images=1010,plan_sha256=sha(OUT/'epoch_plan.json'),initial_sha256=sha(OUT/'initial_state.pth'),test_accessed=False,selection='minimum val loss; full batch1 recheck on every potential best; one epoch per arm',shared_gpu=True,numerical=[],arms=[],timing_note='CUDA events recorded without per-step synchronize; wall includes loader startup, training, batch8 validation, all1010 batch1 best recheck, best/latest save. Diagnostic 3 steps excluded.')
 write('report.json',result)
 model.cuda();crit=api.BceDiceLoss().cuda()
 print('LAB START PID',os.getpid(),'GPU_UUID',subprocess.getoutput('nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader'),flush=True)
 for threads in (4,1):
  workers=2;torch.set_num_threads(threads)
  train=loader(ds,workers,8,plans);val=loader(vd,workers,8);precise=loader(vd,workers,1)
  try:
   # Untimed acceptance: same real first three augmented batches, all parameter gradients and step weights, all buffers.
   reset();opt=optimizer();diag=[];it=iter(train)
   for step in range(3):
    x,y,ids,_=next(it);inp=dict(ids=list(ids),x=sha_bytes(x.numpy().tobytes()),y=sha_bytes(y.numpy().tobytes()))
    opt.zero_grad(set_to_none=True);p=model(x.cuda(non_blocking=True));loss=crit(p,y.cuda(non_blocking=True));loss.backward()
    grads={n:(v.grad.detach().cpu().clone() if v.grad is not None else torch.zeros_like(v,device='cpu')) for n,v in model.named_parameters()}
    missing=[n for n,v in model.named_parameters() if v.grad is None]
    opt.step();weights={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
    payload=dict(input=inp,loss=float(loss.detach()),prediction=p.detach().cpu(),gradients=grads,weights=weights,missing=missing)
    fn=OUT/f'diagnostic_step{step+1}.pth'
    if threads==4:torch.save(payload,fn);diag.append(dict(step=step+1,loss=payload['loss'],missing=missing,finite=all(torch.isfinite(g).all().item() for g in grads.values())))
    else:
     ref=torch.load(fn,map_location='cpu');diag.append(dict(step=step+1,input_equal=ref['input']==inp,loss_abs=abs(payload['loss']-ref['loss']),prediction=compare({'prediction':ref['prediction']},{'prediction':payload['prediction']}),gradients=compare(ref['gradients'],grads),weights_buffers=compare(ref['weights'],weights),missing_equal=ref['missing']==missing))
    del grads,weights,payload
   result['numerical'].append(dict(threads=threads,workers=workers,steps=diag));write('report.json',result)
   if threads==4:
    reset();opt=optimizer();selfdiag=[];it=iter(train)
    for step in range(3):
     x,y,ids,_=next(it);inp=dict(ids=list(ids),x=sha_bytes(x.numpy().tobytes()),y=sha_bytes(y.numpy().tobytes()))
     opt.zero_grad(set_to_none=True);p=model(x.cuda(non_blocking=True));loss=crit(p,y.cuda(non_blocking=True));loss.backward()
     grads={n:(v.grad.detach().cpu().clone() if v.grad is not None else torch.zeros_like(v,device='cpu')) for n,v in model.named_parameters()};opt.step()
     weights={n:v.detach().cpu().clone() for n,v in model.state_dict().items()};ref=torch.load(OUT/f'diagnostic_step{step+1}.pth',map_location='cpu')
     selfdiag.append(dict(step=step+1,input_equal=ref['input']==inp,loss_abs=abs(float(loss.detach())-ref['loss']),prediction=compare({'prediction':ref['prediction']},{'prediction':p.detach().cpu()}),gradients=compare(ref['gradients'],grads),weights_buffers=compare(ref['weights'],weights)))
     del grads,weights,ref
    result['baseline_self_replay']=selfdiag;write('report.json',result)
   # Reset all state after diagnostics; no diagnostic step is reused as training.
   reset();del opt;opt=optimizer();sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=50,eta_min=1e-5)
   # Fresh workers for wall-clock startup symmetry.
   close(train);close(val);close(precise)
   train=loader(ds,workers,8,plans);val=loader(vd,workers,8);precise=loader(vd,workers,1)
   torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();wait=0.;events=[];losses=[];it=iter(train)
   for step in range(1,874):
    t=time.perf_counter();x,y,ids,_=next(it);wait+=time.perf_counter()-t
    ev=[torch.cuda.Event(enable_timing=True) for _ in range(5)];ev[0].record()
    x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);ev[1].record()
    opt.zero_grad(set_to_none=True);p=model(x);loss=crit(p,y);ev[2].record();loss.backward();ev[3].record();opt.step();ev[4].record()
    losses.append(loss.detach());events.append(ev)
    if step==1 or step%100==0 or step==873:
     print('REAL TRAIN workers',workers,'step',step,'/873','loss',float(torch.stack(losses).mean()),flush=True)
     write('progress.json',dict(pid=os.getpid(),workers=workers,step=step,total=873,elapsed=time.perf_counter()-start))
   sched.step();torch.cuda.synchronize();train_wall=time.perf_counter()-start
   phases={name:sum(e[i].elapsed_time(e[i+1]) for e in events)/1000 for i,name in enumerate(['h2d','forward_loss','backward','optimizer'])}
   t=time.perf_counter();validation8=evaluate_fast(model,val,crit,torch.device('cuda'),api);validation=api.evaluate(model,precise,crit,torch.device('cuda'));torch.cuda.synchronize();val_wall=time.perf_counter()-t
   t=time.perf_counter();armout=OUT/f'threads{threads}';armout.mkdir()
   torch.save(model.state_dict(),armout/'best.pth');torch.save(dict(epoch=1,model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),best_epoch=1,best_loss=validation['loss'],python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),plan_sha256=result['plan_sha256']),armout/'latest.pth')
   torch.cuda.synchronize();save_wall=time.perf_counter()-t;wall=time.perf_counter()-start
   arm=dict(threads=threads,workers=workers,completed=True,optimizer_updates=873,train_images=len(ds),validation_images=validation['images'],validation8_images=validation8['images'],wall_seconds=wall,train_wall_seconds=train_wall,validation_wall_seconds=val_wall,save_wall_seconds=save_wall,data_wait_host_seconds=wait,cuda_event_seconds=phases,train_loss=float(torch.stack(losses).double().mean()),validation=validation,validation8_loss=validation8['loss'],peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
   result['arms'].append(arm);write('report.json',result);write(f'threads{threads}/summary.json',arm);print('EPOCH COMPLETE',json.dumps(arm),flush=True)
   if threads==4 and wall<=70:
    result['complete']=True;result['stop_reason']='Slow baseline not reproduced (<=70s); no optimization tested';write('report.json',result);return
  finally:
   for l in (train,val,precise):close(l)
 result['complete']=len(result['arms'])==2
 result['ratio']=result['arms'][0]['wall_seconds']/result['arms'][1]['wall_seconds']
 s4=torch.load(OUT/'threads4/best.pth',map_location='cpu',weights_only=True);s1=torch.load(OUT/'threads1/best.pth',map_location='cpu',weights_only=True)
 result['epoch_weight_buffer_difference']=compare(s4,s1)
 write('report.json',result)
 print('LAB COMPLETE',result['ratio'],flush=True)
def sha_bytes(b):return hashlib.sha256(b).hexdigest()
if __name__=='__main__':
 if '--child' in sys.argv:
  try:child_main()
  except BaseException as e:write('failure.json',dict(error=repr(e),traceback=traceback.format_exc(),pid=os.getpid()));raise
 else:controller()

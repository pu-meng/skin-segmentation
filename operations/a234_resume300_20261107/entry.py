import os,sys,json,csv,time,hashlib,shutil,subprocess,fcntl,importlib.util,inspect,random,copy
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation'); OPS=ROOT/'operations/a234_resume300_20261107'; BASE=Path('/media/8T/PuMengYu/experiments/skin_segmentation'); RUN=BASE/'A234_RESUME100to300_20261107'
JOBS={'A2':('DualA_E2_FirstBatch100_20261107','Input','DInput'),'A3':('DualA_E2_Cross_AFNO_GPU1_20261107','Cross','DCross'),'A4':('DualA_E2_Cross_AFNO_GPU1_20261107','AFNO','DAFNO')}
UUIDS=['GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde','GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b']; PY='/home/PuMengYu/anaconda3/envs/medseg/bin/python'
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2));t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def builder():
 sys.path[:0]=[str(ROOT/'operations/duala_E2_batch1_20261107'),str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
 spec=importlib.util.spec_from_file_location('a234_original_builder',ROOT/'operations/duala_E2_batch1_20261107/entry.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def equal(a,b):
 import torch,numpy as np
 if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
 if isinstance(a,np.ndarray):return np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b

def paths(job):
 folder,arm,_=JOBS[job];return BASE/folder,BASE/folder/arm/'run',RUN/job/'run'
def verify(job):
 old,src,dst=paths(job);c=json.loads((RUN/'protocol.json').read_text());p=json.loads((RUN/job/'protocol.json').read_text())
 assert sha(__file__)==c['entry_sha256'],'entry drift'
 for f,h in c['protected_sha256'].items():assert sha(f)==h,'protected drift '+f
 assert all(shutil.disk_usage(x).free>8*2**30 for x in [ROOT,BASE]),'disk below 8GiB'
 assert sha(p['config']['data']['split_file'])==p['config']['data']['split_sha256']
 return p

def restore_check(job,model,opt,scheduler,g,loaders,p,phase,gpu=None):
 import torch,numpy as np
 old,src,dst=paths(job);cfg=verify(job)['config'];rows=list(csv.DictReader((dst/'history.csv').open()));win=min(rows,key=lambda r:float(r['val_loss']))
 assert p['epoch']==100 and [int(r['epoch']) for r in rows]==list(range(1,101))
 assert p['protocol_sha256']==sha(RUN/job/'protocol.json') and p['split_sha256']==cfg['data']['split_sha256']
 assert int(win['epoch'])==p['best_epoch'] and float(win['val_loss'])==p['best_loss']
 assert sha(dst/'checkpoints/best.pth')==sha(src/'checkpoints/best.pth')
 assert equal(model.state_dict(),p['model']) and equal(opt.state_dict(),p['optimizer']) and equal(scheduler.state_dict(),p['scheduler'])
 assert scheduler.T_max==50 and scheduler.last_epoch==100
 assert equal(random.getstate(),p['python_rng']) and equal(np.random.get_state(),p['numpy_rng']) and equal(torch.get_rng_state(),p['torch_rng']) and equal(g.get_state(),p['loader_rng'])
 assert loaders['train'].generator is g and loaders['train'].loader.sampler.sampler.generator is g
 assert p['cuda_rng'].dtype==torch.uint8 and p['cuda_rng'].numel()>0
 if gpu is not None:assert equal(torch.cuda.get_rng_state(),p['cuda_rng'])
 write(OPS/job/(phase+'_restore_audit.json'),dict(passed=True,pid=os.getpid(),source_checkpoint=str(src/'checkpoints/latest.pth'),source_checkpoint_sha256=sha(src/'checkpoints/latest.pth'),epoch=100,optimizer_updates=87300,model_strict_tensor_equal=True,optimizer_all_state_equal=True,scheduler=p['scheduler'],python_numpy_torch_sampler_rng_equal=True,cuda_rng_restored_and_equal=gpu is not None,cuda_rng_validated_cpu_only=gpu is None,loader_worker_rng='legacy checkpoint has no worker generator snapshot; workers __getitem__ deterministic, augmentation Python/NumPy in parent, sampler and EpochLoader share restored generator; worker seeds do not affect examples',history_rows=100,best_epoch=p['best_epoch'],best_loss=p['best_loss'],gpu_uuid=None if gpu is None else UUIDS[gpu]))
 print('RESTORE_ACCEPTED',job,phase,'epoch100 updates87300 T_max50',flush=True)

def prepare():
 assert not RUN.exists(),'existing run: inspect, do not duplicate'
 OPS.mkdir(exist_ok=True);RUN.mkdir();m=builder();protected={};records={}
 for job,(folder,arm,closeout) in JOBS.items():
  old,src,dst=paths(job);p=json.loads((old/'protocol.json').read_text());rows=list(csv.DictReader((src/'history.csv').open()));assert len(rows)==100
  for f,h in {**p['protected_sha256'],**p['implementation_sha256']}.items():assert sha(f)==h,'source drift '+f;protected[f]=h
  for f in [old/'protocol.json',src/'history.csv',src/'checkpoints/latest.pth',src/'checkpoints/best.pth',BASE/'Closeout_Test6_20261107'/closeout/'completion.json']:protected[str(f)]=sha(f)
  dst.mkdir(parents=True);(dst/'checkpoints').mkdir();shutil.copy2(old/'protocol.json',RUN/job/'protocol.json')
  for name in ['history.csv','status.json','provenance.json']:shutil.copy2(src/name,dst/name)
  (dst/'checkpoints/latest.pth').symlink_to(src/'checkpoints/latest.pth');shutil.copy2(src/'checkpoints/best.pth',dst/'checkpoints/best.pth')
  records[job]=dict(original_run=str(src),resume_run=str(dst),arm=arm,display=closeout,old_protocol_sha256=sha(old/'protocol.json'),latest_sha256=sha(src/'checkpoints/latest.pth'),old_test=str(BASE/'Closeout_Test6_20261107'/closeout))
 write(RUN/'protocol.json',dict(identity='A2 Input / A3 Cross / A4 AFNO latest100 to total300',jobs=records,start_epoch=100,total_epochs=300,updates_per_epoch=873,total_optimizer_updates=261900,scheduler_T_max=50,scheduler_evidence='all three actual latest.scheduler.T_max=50 last_epoch=100, not1000',batch=8,last_batch=7,precision='FP32',loader_workers=2,validation_fast_batch=8,validation_selection_recheck_batch=1,selection='earliest minimum val_loss across preserved100 + appended200',test='same selected best none + mirror4, 2020 exact IDs each, metrics CSV summary 12 visualizations provenance SHA',protected_sha256=protected,entry_sha256=sha(__file__),deadline_seconds=259200,priority='explicit parent dispatch A1+SkipC fixed GPU1 first; A2/A3/A4 fixed GPU0 sequential, no GPU1 completion dependency; never share occupied GPU0',disk_guard_gib=8,no_deletion=True))
 print('PREPARED',RUN,flush=True)

def cpu(job):
 import torch,numpy as np
 m=builder();torch.set_num_threads(1);p0=verify(job);model=m.make(JOBS[job][1],False);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8);s=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=50,eta_min=1e-5);loaders,g=m.make_fast_loaders(p0['config'],m.runner.baseline_api(),workers=2,validation_batch=8)
 p=torch.load(paths(job)[1]/'checkpoints/latest.pth',map_location='cpu',weights_only=False);model.load_state_dict(p['model'],strict=True);opt.load_state_dict(p['optimizer']);s.load_state_dict(p['scheduler']);random.setstate(p['python_rng']);np.random.set_state(p['numpy_rng']);torch.set_rng_state(p['torch_rng']);g.set_state(p['loader_rng']);restore_check(job,model,opt,s,g,loaders,p,'cpu');assert not torch.cuda.is_initialized()
 # CPU optimizer/state and scheduler serialized round trip, without data iteration or updates.
 import io
 out=io.BytesIO();torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':s.state_dict(),'rng':g.get_state()},out);out.seek(0);q=torch.load(out,map_location='cpu',weights_only=False);assert equal(q['model'],p['model']) and equal(q['optimizer'],p['optimizer']) and equal(q['scheduler'],p['scheduler']) and equal(q['rng'],p['loader_rng'])
 write(OPS/job/'cpu_roundtrip.json',dict(passed=True,cuda_initialized=False,actual_checkpoint=True,strict_model=True,all_optimizer_tensors=True,scheduler=True,sampler_rng=True,updates_performed=0))

def gpu_phase(job,gpu,phase):
 import torch
 m=builder();torch.set_num_threads(4);torch.cuda.set_device(0);p0=verify(job);cfg=copy.deepcopy(p0['config']);cfg['training'].update(epochs=300,total_optimizer_updates=261900,image_exposures=300*cfg['data']['counts']['train']);api=m.runner.baseline_api();api.seed_all(42)
 model=m.make(JOBS[job][1],False).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8);loaders,g=m.make_fast_loaders(cfg,api,workers=2,validation_batch=8);criterion=api.BceDiceLoss().cuda();dst=paths(job)[2]
 mapping=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True);assert any(UUIDS[gpu] in l and f', {os.getpid()}' in l for l in mapping.splitlines())
 try:
  if phase=='preflight':
   import numpy as np
   s=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=50,eta_min=1e-5);p=torch.load(dst/'checkpoints/latest.pth',map_location='cpu',weights_only=False);model.load_state_dict(p['model'],strict=True);opt.load_state_dict(p['optimizer']);s.load_state_dict(p['scheduler']);random.setstate(p['python_rng']);np.random.set_state(p['numpy_rng']);torch.set_rng_state(p['torch_rng']);torch.cuda.set_rng_state(p['cuda_rng']);g.set_state(p['loader_rng']);restore_check(job,model,opt,s,g,loaders,p,phase,gpu);del p
   model.train();torch.cuda.reset_peak_memory_stats();x,y,ids,_=next(iter(loaders['train']));assert list(x.shape)==[8,3,256,256];opt.zero_grad(set_to_none=True);loss=criterion(model(x.cuda()),y.cuda());loss.backward();grad=[p.grad for p in model.parameters() if p.grad is not None];assert bool(torch.stack([torch.isfinite(t).all() for t in grad]).all());param=model.decoder.e2.gamma;before=param.detach().clone();opt.step();torch.cuda.synchronize();assert not torch.equal(before,param)
   write(OPS/job/'preflight_passed.json',dict(passed=True,pid=os.getpid(),gpu_uuid=UUIDS[gpu],batch=list(x.shape),loss=loss.item(),gamma_delta=(param-before).abs().max().item(),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,preflight_discarded=True));print('REAL_BATCH8_PREFLIGHT_PASSED',job,flush=True)
  else:
   assert json.loads((OPS/job/'preflight_passed.json').read_text())['passed'];original=opt.step;seen=False
   import types
   def step(self,*a,**kw):
    nonlocal seen
    if seen:return original(*a,**kw)
    before=model.decoder.e2.gamma.detach().clone();result=original(*a,**kw);torch.cuda.synchronize();assert not torch.equal(before,model.decoder.e2.gamma);seen=True;write(OPS/job/'first_resumed_optimizer_update.json',dict(pid=os.getpid(),gpu_uuid=UUIDS[gpu],epoch=101,step=1,optimizer_updates=87301,gamma_delta=(before-model.decoder.e2.gamma).abs().max().item(),total_budget=300));print('FORMAL_RESUMED_UPDATE',job,87301,flush=True);return result
   opt.step=types.MethodType(step,opt)
   # Reuse original runner training body; isolate only old identity lookup + insert restore audit and disk/source checks.
   source=inspect.getsource(m.runner.train);source=source.replace('        from recover_ham_dedup import inspect_run\n        record = inspect_run(provenance[\'model\'], config)','        record = dict(isolated_resume=True)')
   needle="        generator.set_state(payload['loader_rng'])";assert needle in source;source=source.replace(needle,needle+"\n        audit_restore(payload, model, optimizer, scheduler, generator, loaders)")
   source=source.replace('    for epoch in range(start_epoch, epochs + 1):','    for epoch in range(start_epoch, epochs + 1):\n        verify_boundary()')
   namespace=dict(m.runner.__dict__);namespace.update(audit_restore=lambda p,mod,o,s,g,l:restore_check(job,mod,o,s,g,l,p,'train',gpu),verify_boundary=lambda:verify(job));exec(compile(source,str(OPS/'isolated_original_runner_train'), 'exec'),namespace)
   info=dict(model=JOBS[job][2]+' '+job+' RESUME300',protocol_sha256=sha(RUN/job/'protocol.json'),engineering=dict(resume_protocol=str(RUN/'protocol.json'),old_run=str(paths(job)[1])))
   namespace['train'](model,loaders,criterion,opt,torch.device('cuda:0'),dst,cfg,info,g,resume=True)
   assert json.loads((dst/'artifact_audit.json').read_text())['epochs']==300;write(OPS/job/'train_complete.json',dict(complete=True,epochs=300))
 finally:
  for loader in loaders.values():
   if hasattr(loader,'close'):loader.close()

def test(job,gpu):
 import torch
 m=builder();torch.set_num_threads(2);p0=verify(job);old,src,dst=paths(job);rows=list(csv.DictReader((dst/'history.csv').open()));assert [int(r['epoch']) for r in rows]==list(range(1,301));win=min(rows,key=lambda r:float(r['val_loss']));p=torch.load(dst/'checkpoints/latest.pth',map_location='cpu',weights_only=False);assert p['epoch']==300 and p['best_epoch']==int(win['epoch']) and p['best_loss']==float(win['val_loss']);del p
 work=RUN/job/'final_test';work.mkdir();snap=work/'fixed_best.pth';cp=dst/'checkpoints/best.pth';h=sha(cp);shutil.copy2(cp,snap);assert h==sha(snap)==sha(cp);model=m.make(JOBS[job][1],False);model.load_state_dict(torch.load(snap,map_location='cpu',weights_only=False),strict=True);assert not torch.cuda.is_initialized();write(work/'cpu_load_audit.json',dict(strict=True,checkpoint_sha256=h,selected_epoch=int(win['epoch']),cuda_initialized=False))
 from torch.utils.data import DataLoader
 import final_test_pipeline as pipeline
 from operations.nnunet_mirror_validation.run_validation import Predictor
 api=m.runner.baseline_api();api.seed_all(42);ds=api.HAMDataset(Path(p0['config']['data']['split_file']).parent,'test');ids={r['image_id'] for r in ds.rows};assert len(ids)==len(ds)==2020 and not ds.training;loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=0);model.cuda().eval();summaries={}
 for mode in ['none','mirror4']:
  target=work/mode;assert not target.exists();pred=Predictor(model,mode=='mirror4').eval();print('START_TEST',job,mode,os.getpid(),flush=True);result=api.evaluate(pred,loader,api.BceDiceLoss().cuda(),torch.device('cuda:0'),target);assert len(pred.times)==2020 and pred.verify_remaining==0
  result.update(split='test',dataset='HAM10000',name=job+' '+JOBS[job][2]+' RESUME300',selected_epoch=int(win['epoch']),checkpoint=str(snap),checkpoint_sha256=h,split_sha256=p0['config']['data']['split_sha256'],tta=mode=='mirror4',training_epochs=300,test_used_for_checkpoint_selection=False,physical_gpu=gpu)
  a=pipeline.validate_outputs(target,ids,result);assert a['complete'];write(target/'summary.json',result);write(target/'artifact_audit.json',a);pred.close();summaries[mode]=result['global'];print('TEST_COMPLETE',job,mode,flush=True)
 verify(job);write(OPS/job/'completion.json',dict(complete=True,training_epochs=300,test_images_per_mode=2020,modes=['none','mirror4'],checkpoint_sha256=h,selected_epoch=int(win['epoch']),results=summaries,result_root=str(work)))

def compute():
 lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines();return [(l.split(',')[0].strip(),int(l.split(',')[1])) for l in lines if l.strip()]

def controller():
 with (OPS/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);deadline=time.time()+259200;state=dict(pid=os.getpid(),deadline_unix=deadline,status='waiting_priority_A1',completed=[]);write(OPS/'queue_status.json',state);gpu=None
  while time.time()<deadline:
   apps=compute();reason='fixed GPU0 has existing compute PID; no sharing authorized'
   if not any(u==UUIDS[0] for u,p in apps):
    gpu=0;write(OPS/'priority_accepted.json',dict(parent_dispatch='A1 SkipC fixed GPU1 first, verified actual compute PID965993 on GPU1 before deployment; A234 fixed GPU0, no cross-card dependency',observed_compute=apps,verified_at=time.time(),our_gpu=gpu));break
   state.update(status='waiting_priority_or_gpu',condition=reason,compute=apps,updated_unix=time.time());write(OPS/'queue_status.json',state);print('WAIT',reason,'compute',apps,flush=True);time.sleep(30)
  assert gpu is not None,'priority/resource deadline expired'
  with (ROOT/'operations'/f'a234_gpu{gpu}.lock').open('a') as gpu_lock:
   fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   for job in JOBS:
    for phase in ['preflight','train','test']:
     assert time.time()<deadline,'queue deadline';verify(job)
     assert not any(u==UUIDS[gpu] for u,p in compute()),'GPU gained unrelated compute process'
     env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=UUIDS[gpu],EXPERIMENT_PHYSICAL_GPU=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
     child=subprocess.Popen([PY,'-B','-u',str(OPS/'entry.py'),phase,job,str(gpu)],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1);state.update(status=phase,job=job,gpu=gpu,gpu_uuid=UUIDS[gpu],child_pid=child.pid);write(OPS/'queue_status.json',state)
     with (OPS/job/(phase+'.log')).open('x') as log:
      for line in child.stdout:print(line,end='',flush=True);log.write(line);log.flush()
     rc=child.wait();write(OPS/job/(phase+'_exitcode.json'),dict(exitcode=rc));assert rc==0,f'{job} {phase} failed {rc}'
    assert json.loads((OPS/job/'completion.json').read_text())['complete'];state['completed'].append(job)
   state.update(status='complete');write(OPS/'queue_status.json',state)

if __name__=='__main__':
 phase=sys.argv[1]
 try:
  if phase=='prepare':prepare()
  elif phase=='cpu':cpu(sys.argv[2])
  elif phase=='controller':controller()
  elif phase=='test':test(sys.argv[2],int(sys.argv[3]))
  else:gpu_phase(sys.argv[2],int(sys.argv[3]),phase)
 except BaseException as e:
  write(OPS/('failure_'+phase+'.json'),dict(error=repr(e),pid=os.getpid(),phase=phase));raise

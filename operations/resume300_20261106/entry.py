"""Isolated resume100->300 and fixed-val-best no-TTA test relay."""
import os,sys,json,time,csv,shutil,subprocess,importlib,random
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
ARM=os.environ['RESUME_ARM']; GROUP='duala' if ARM=='DualA' else 'fourarm'
OPS=ROOT/'operations/resume300_20261106'
OLD=(Path('/media/8T/PuMengYu/experiments/skin_segmentation/DecoderDualA_SCREEN100of300_20261106') if ARM=='DualA' else ROOT/'log/TinyViM_E1232_FourArm_SCREEN100of300_20261106')
RUN=Path('/media/8T/PuMengYu/experiments/skin_segmentation')/('DecoderDualA_RESUME100to300_20261106' if ARM=='DualA' else 'TinyViM_E1232_FourArm_RESUME100to300_20261106')
m=importlib.import_module('source.'+('decoder_dual_a_20261106' if ARM=='DualA' else 'e1232_fourarm_screen_20261106')+'.entry')
m.RUN=RUN
import torch
import numpy as np
PATH=RUN/ARM/'train'; EVID=OPS/ARM; EVID.mkdir(exist_ok=True)
def write(name,value):m.write(EVID/name,value)
def barrier(suffix):
 peers=['DualA'] if GROUP=='duala' else ['R0','R1','R2','R3']
 deadline=time.monotonic()+1800
 while not all((OPS/a/suffix).exists() for a in peers):
  assert not any((OPS/a/'failure.json').exists() for a in peers),'peer failed'
  if time.monotonic()>deadline:raise TimeoutError(suffix)
  time.sleep(1)
def equal(a,b):
 if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
 if isinstance(a,np.ndarray):return np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b
orig_restore=m.restore
PHASE=sys.argv[1]
def audited_restore(p,model,opt,scheduler,generator,loaders):
 orig_restore(p,model,opt,scheduler,generator,loaders)
 assert p['epoch']==100 and p['optimizer_updates']==87300
 assert equal(model.state_dict(),p['model']) and equal(opt.state_dict(),p['optimizer'])
 assert equal(scheduler.state_dict(),p['scheduler']) and scheduler.T_max==50
 assert equal(random.getstate(),p['python_rng']) and equal(np.random.get_state(),p['numpy_rng'])
 assert equal(torch.get_rng_state(),p['torch_rng']) and equal(torch.cuda.get_rng_state_all(),p['cuda_rng'])
 assert equal(generator.get_state(),p['loader_rng']) and equal(m.loader_streams(loaders),p['loader_generators'])
 rows=list(csv.DictReader((PATH/'history.csv').open()));assert len(rows)==100
 win=min(rows,key=lambda r:float(r['val_loss']))
 assert int(win['epoch'])==p['best_epoch'] and float(win['val_loss'])==p['best_loss']
 assert m.sha256_file(PATH/'checkpoints/best.pth')==m.sha256_file(OLD/ARM/'train/checkpoints/best.pth')
 write(PHASE+'_restore_audit.json',dict(passed=True,pid=os.getpid(),gpu_uuid=m.UUID,epoch=100,updates=87300,best_epoch=p['best_epoch'],best_loss=p['best_loss'],scheduler=scheduler.state_dict(),lr=opt.param_groups[0]['lr'],model_strict_and_tensor_equal=True,optimizer_all_state_tensor_equal=True,python_numpy_torch_cuda_loader_rng_equal=True,loader_worker_randomness='parent planned; deterministic worker __getitem__',source_checkpoint=str(OLD/ARM/'train/checkpoints/latest.pth'),source_checkpoint_sha256=m.sha256_file(OLD/ARM/'train/checkpoints/latest.pth'),history_rows=100,protocol_sha256=m.sha256_file(RUN/'protocol.json')))
 print('RESTORE100_ACCEPTED',ARM,PHASE,'PID',os.getpid(),'UUID',m.UUID,flush=True)
 if PHASE=='train':
  write('formal_ready.json',dict(pid=os.getpid()));barrier('formal_ready.json')
m.restore=audited_restore

def training():
 config=m.verify()['config'];api=m.runner.baseline_api();api.seed_all(42);torch.set_num_threads(4);torch.cuda.set_device(0)
 model=(m.build_candidate(True) if ARM=='DualA' else m.build_candidate(ARM,True)).cuda()
 opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8)
 scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=50,eta_min=1e-5)
 loaders,generator=m.make_fast_loaders(config,api,workers=2,validation_batch=8);criterion=api.BceDiceLoss().cuda()
 mapping=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
 assert any(m.UUID in line and f', {os.getpid()},' in line for line in mapping.splitlines()),mapping
 write(PHASE+'_identity.json',dict(pid=os.getpid(),gpu_uuid=m.UUID,run=str(RUN),arm=ARM,batch=8,precision='FP32',model_parameters=sum(p.numel() for p in model.parameters()),phase=PHASE))
 try:
  if PHASE=='preflight':
   p=torch.load(PATH/'checkpoints/latest.pth',map_location='cpu',weights_only=False);m.restore(p,model,opt,scheduler,generator,loaders);del p
   write('resident_ready.json',dict(pid=os.getpid()));barrier('resident_ready.json')
   torch.cuda.reset_peak_memory_stats();model.train();x,y,ids,_=next(iter(loaders['train']))
   opt.zero_grad(set_to_none=True);loss=criterion(model(x.cuda()),y.cuda());loss.backward();m.finite_gradients(model)
   name,param=next((n,p) for n,p in model.named_parameters() if p.grad is not None and bool(p.grad.abs().max()>0))
   before=param.detach().clone();opt.step();torch.cuda.synchronize();assert not torch.equal(before,param)
   write('preflight_passed.json',dict(passed=True,pid=os.getpid(),gpu_uuid=m.UUID,real_batch=list(x.shape),loss=loss.item(),parameter=name,numerical_update=True,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,checkpoint_untouched=True,preflight_weights_discarded=True))
   print('RESUMED_REAL_BATCH8_PREFLIGHT_PASS',ARM,flush=True);barrier('preflight_passed.json')
  else:
   original_step=opt.step;seen=False
   def step(*a,**kw):
    nonlocal seen
    if seen:return original_step(*a,**kw)
    name,param=next((n,p) for n,p in model.named_parameters() if p.grad is not None and bool(p.grad.abs().max()>0));before=param.detach().clone()
    result=original_step(*a,**kw);torch.cuda.synchronize();changed=not torch.equal(before,param);assert changed
    write('first_resumed_optimizer_update.json',dict(pid=os.getpid(),gpu_uuid=m.UUID,epoch=101,step=1,optimizer_updates=87301,parameter=name,numerical_update=True,max_parameter_delta=float((before-param).abs().max()),total_budget=300))
    print('FORMAL_RESUMED_OPTIMIZER_UPDATE',ARM,87301,flush=True);seen=True;return result
   opt.step=step
   m.train(ARM,model,opt,scheduler,loaders,generator,criterion,torch.device('cuda:0'),PATH,resume=True,stop=300)
 finally:
  for loader in loaders.values():
   if hasattr(loader,'close'):loader.close()

def final_test():
 import final_test_pipeline as pipeline
 out=RUN/'final_test'/ARM;out.mkdir(parents=True,exist_ok=True)
 rows=list(csv.DictReader((PATH/'history.csv').open()));assert [int(r['epoch']) for r in rows]==list(range(1,301))
 s=json.loads((PATH/'status.json').read_text());assert s['epoch_completed']==300 and s['status']=='budget_complete' and s['optimizer_updates']==261900
 p=torch.load(PATH/'checkpoints/latest.pth',map_location='cpu',weights_only=False);winner=min(rows,key=lambda r:float(r['val_loss']))
 assert p['epoch']==300 and p['best_epoch']==int(winner['epoch']) and p['best_loss']==float(winner['val_loss']);del p
 cp=PATH/'checkpoints/best.pth';snapshot=out/'fixed_best.pth';assert not snapshot.exists()
 h=m.sha256_file(cp);shutil.copy2(cp,snapshot);assert h==m.sha256_file(cp)==m.sha256_file(snapshot)
 model=(m.build_candidate(False) if ARM=='DualA' else m.build_candidate(ARM,False));model.load_state_dict(torch.load(snapshot,map_location='cpu',weights_only=False),strict=True)
 fixed=dict(checkpoint=str(snapshot),checkpoint_sha256=h,selected_epoch=int(winner['epoch']),history_rows=300,split_sha256=m.manifest()['config']['data']['split_sha256'],seed=42,parameters=sum(p.numel() for p in model.parameters()),selection='earliest minimum validation loss; no test selection')
 pipeline.ARMS={ARM:(ARM,PATH)};pipeline.build_model=lambda _:model
 pipeline.training_ready=lambda _:True;pipeline.fixed_best_record=lambda _:fixed
 pipeline.aggregate=lambda _:None
 # Isolate test loading from multiprocessing import state; metrics/preprocessing unchanged.
 import torch.utils.data
 real_loader=torch.utils.data.DataLoader
 def loader(*a,**kw):kw['num_workers']=0;kw.pop('multiprocessing_context',None);return real_loader(*a,**kw)
 torch.utils.data.DataLoader=loader
 sys.argv=['final_test','--arm',ARM,'--output-root',str(RUN/'final_test'),'--gpu','0']
 pipeline.main()
 assert json.loads((out/'artifact_audit.json').read_text())['complete']
 write('completion.json',dict(training_complete=True,epochs=300,test_complete=True,test_images=2020,tta=False,final_test=str(out),checkpoint_sha256=h))

if __name__=='__main__':
 try:
  if PHASE=='test':final_test()
  else:training()
 except BaseException as e:write('failure.json',dict(error=repr(e),phase=PHASE,pid=os.getpid()));raise

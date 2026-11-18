"""Independent validation-only screening; latest epoch-boundary continuation supported."""
import os,sys,json,random,time,csv,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
from pretrained_unet2d import prepare_imports,sha256_file
prepare_imports('mambavision_t');prepare_imports('tinyvim_s')
import numpy as np
import torch
import ham_fast_runner as runner
from ham_fast_pipeline import make_fast_loaders,evaluate_fast
from source.decoder_dual_a_20261106.candidate import build_candidate
ARMS={"DualA":"Candidate A; D2 Mamba1 + RepDW7 replacement"}
RUN=Path('/media/8T/PuMengYu/experiments/skin_segmentation/DecoderDualA_SCREEN100of300_20261106')
UUID='GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b'
def write(p,o):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);runner.write_json(p,o)
def manifest():return json.loads((RUN/'protocol.json').read_text())
def verify():
 m=manifest()
 for p,h in m['source_sha256'].items():assert sha256_file(ROOT/p)==h,'source drift '+p
 assert sha256_file(m['config']['data']['split_file'])==m['config']['data']['split_sha256']
 assert shutil.disk_usage(RUN).free>8*2**30,'disk below 8GiB'
 return m

def loader_streams(loaders):
 return {k:(getattr(v,'loader',v)).generator.get_state() for k,v in loaders.items()}
def capture(model,opt,scheduler,generator,loaders,epoch,best_loss,best_epoch,updates):
 assert loaders['train'].generator is generator
 assert loaders['train'].loader.sampler.sampler.generator is generator
 return dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,best_loss=best_loss,best_epoch=best_epoch,optimizer_updates=updates,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),loader_rng=generator.get_state(),loader_generators=loader_streams(loaders),protocol_sha256=sha256_file(RUN/'protocol.json'),resume_boundary='completed epoch; sampler exhausted; parent-planned augmentation; workers deterministic __getitem__')
def restore(p,model,opt,scheduler,generator,loaders):
 assert p['protocol_sha256']==sha256_file(RUN/'protocol.json')
 model.load_state_dict(p['model'],strict=True);opt.load_state_dict(p['optimizer']);scheduler.load_state_dict(p['scheduler'])
 random.setstate(p['python_rng']);np.random.set_state(p['numpy_rng']);torch.set_rng_state(p['torch_rng']);torch.cuda.set_rng_state_all(p['cuda_rng']);generator.set_state(p['loader_rng'])
 for k,v in loaders.items():getattr(v,'loader',v).generator.set_state(p['loader_generators'][k])
 assert scheduler.last_epoch==p['epoch']

def wait_marker(name,seconds=1800):
 deadline=time.monotonic()+seconds
 while not (RUN/name).exists():
  if (RUN/'FAILED.json').exists():raise RuntimeError('peer/controller failed; release blocked')
  if time.monotonic()>deadline:raise TimeoutError(name)
  time.sleep(.5)

def finite_gradients(model):
 gradients=[p.grad for p in model.parameters() if p.grad is not None]
 assert gradients and bool(torch.stack([torch.isfinite(g).all() for g in gradients]).all()),'nonfinite gradients'
 missing=[n for n,p in model.named_parameters() if p.requires_grad and p.grad is None];assert not missing,missing

def preflight(arm,model,opt,scheduler,loaders,generator,criterion,device,path):
 model.train();torch.cuda.reset_peak_memory_stats();wait_marker('PREFLIGHT_GO')
 rows=[]
 for step,(x,y,ids,_) in enumerate(loaders['train'],1):
  opt.zero_grad(set_to_none=True);pred=model(x.to(device));loss=criterion(pred,y.to(device));assert pred.shape==(8,1,256,256) and bool(torch.isfinite(loss));loss.backward();finite_gradients(model);opt.step();torch.cuda.synchronize()
  rows.append(dict(step=step,loss=loss.item(),image_ids=list(ids)));print('REAL_BATCH8_OPTIMIZER_STEP',arm,rows[-1],flush=True)
  if step==3:break
 # True model+AdamW+scheduler+all RNG checkpoint strict load, including actual loader streams.
 payload=capture(model,opt,scheduler,generator,loaders,0,float('inf'),0,3)
 p=path/'resume_roundtrip.pth';torch.save(payload,p);loaded=torch.load(p,map_location='cpu',weights_only=False)
 restore(loaded,model,opt,scheduler,generator,loaders)
 assert torch.equal(generator.get_state(),loaded['loader_rng'])
 assert all(torch.equal(v.cpu(),loaded['model'][k]) for k,v in model.state_dict().items())
 assert opt.state_dict()['param_groups']==loaded['optimizer']['param_groups']
 assert all(torch.equal(v.cpu(),loaded['optimizer']['state'][k][n]) for k,state in opt.state_dict()['state'].items() for n,v in state.items() if torch.is_tensor(v))
 # Keep real 3-step roundtrip checkpoint as preflight evidence.
 write(path/'preflight.json',dict(passed=True,pid=os.getpid(),gpu_uuid=UUID,batch=8,precision='fp32/no AMP',steps=rows,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,strict_resume_roundtrip=True,preflight_weights_discarded=True,test_data_accessed=False))
 # Keep every arm's model+optimizer allocations resident until all peers passed.
 print('SINGLE_CANDIDATE_PREFLIGHT_PASS',arm,flush=True)

def train(arm,model,opt,scheduler,loaders,generator,criterion,device,path,resume=False,stop=100):
 cp=path/'checkpoints';cp.mkdir(exist_ok=True);best_loss=float('inf');best_epoch=0;updates=0;start=1
 if resume:
  p=torch.load(cp/'latest.pth',map_location='cpu',weights_only=False);assert p['optimizer_updates']==p['epoch']*873;restore(p,model,opt,scheduler,generator,loaders);best_loss=p['best_loss'];best_epoch=p['best_epoch'];updates=p['optimizer_updates'];start=p['epoch']+1
 else:assert not (cp/'latest.pth').exists() and not (path/'history.csv').exists(),'fresh refuses existing training'
 for epoch in range(start,stop+1):
  verify();started=time.monotonic();model.train();losses=[];lr=opt.param_groups[0]['lr'];diag={};b=model.decoder.e2
  # Only first batch each epoch: gate distribution and correction norm; gradients one reduction.
  handles=[];gate_samples={}
  for key,mod in [('skip_gs',model.decoder.e3.gs),('skip_gd',model.decoder.e3.gd)]:
   def hook(mod,ins,out,key=key):
    v=2*out.detach().sigmoid();gate_samples[key]=torch.stack([v.min(),v.mean(),v.max()])
   handles.append(mod.register_forward_hook(hook))
  state=dict(status='training',pid=os.getpid(),gpu_uuid=UUID,epoch=epoch,epoch_completed=epoch-1,screen_stop_epoch=stop,scheduler_total_budget=300,test_data_accessed=False)
  write(path/'status.json',state)
  for step,(x,y,ids,_) in enumerate(loaders['train'],1):
   opt.zero_grad(set_to_none=True);pred=model(x.to(device,non_blocking=True));loss=criterion(pred,y.to(device,non_blocking=True));loss.backward()
   if step==1:
    finite_gradients(model)
    diag={k:v.cpu().tolist() for k,v in gate_samples.items()}
    diag['gradient_l2']=torch.stack([p.grad.detach().float().square().sum() for p in model.parameters() if p.grad is not None]).sum().sqrt().item()
    for h in handles:h.remove()
   if updates==0: first_gain_before=model.decoder.stages[2].raw_m.detach().clone()
   opt.step();losses.append(loss.detach());updates+=1
   if updates==1:
    torch.cuda.synchronize();first_gain_after=model.decoder.stages[2].raw_m.detach().clone();assert not torch.equal(first_gain_before,first_gain_after),'no numeric update';write(path/'first_optimizer_step.json',dict(pid=os.getpid(),gpu_uuid=UUID,optimizer_updates=updates,new_mamba_gain_raw_before=first_gain_before.item(),new_mamba_gain_raw_after=first_gain_after.item(),epoch=epoch,step=step,loss=loss.item(),image_ids=list(ids),batch=list(x.shape),precision='FP32/no AMP',protocol_sha256=sha256_file(RUN/'protocol.json')));print('FIRST_FORMAL_OPTIMIZER_UPDATE',updates,'PID',os.getpid(),'UUID',UUID,flush=True)
   if step==1 or step%50==0 or step==len(loaders['train']):
    vals=torch.stack(losses).cpu().double().numpy();assert np.isfinite(vals).all()
    state.update(step=step,steps_per_epoch=len(loaders['train']),optimizer_updates=updates,train_loss=float(vals.mean()));write(path/'status.json',state)
    print(f'TRAIN {arm} epoch={epoch}/300 screen_stop={stop} step={step}/{len(loaders["train"])} loss={vals.mean():.6f} lr={lr:.8f}',flush=True)
  scheduler.step();val=evaluate_fast(model,loaders['val'],criterion,device,runner.baseline_api());rechecked=False
  if val['loss']<=best_loss+.001:val=runner.baseline_api().evaluate(model,loaders['val_precise'],criterion,device);rechecked=True
  if val['loss']<best_loss:
   best_loss=val['loss'];best_epoch=epoch;torch.save(model.state_dict(),cp/'best.pth')
  p=capture(model,opt,scheduler,generator,loaders,epoch,best_loss,best_epoch,updates)
  torch.save(p,cp/'latest.tmp.pth');(cp/'latest.tmp.pth').replace(cp/'latest.pth')
  row=dict(epoch=epoch,train_loss=float(vals.mean()),val_loss=val['loss'],val_global_dice=val['global']['dice'],val_global_iou=val['global']['iou'],lr=lr,seconds=time.monotonic()-started,best_epoch=best_epoch,optimizer_updates=updates,precise_best_recheck=rechecked)
  with (path/'history.csv').open('a',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(row))
   if epoch==1:w.writeheader()
   w.writerow(row)
  write(path/'diagnostics'/f'epoch_{epoch:03d}.json',diag)
  state.update(epoch_completed=epoch,best_epoch=best_epoch,best_loss=best_loss,optimizer_updates=updates);write(path/'status.json',state);print('EPOCH COMPLETE',arm,json.dumps(row),flush=True)
 # Do not load best into live training object; latest remains exact epoch boundary state.
 state.update(status='paused_at_screen_boundary' if stop==100 else 'budget_complete',next_epoch=stop+1,resume_from=str(cp/'latest.pth'),automatic_continuation=False,test_data_accessed=False)
 write(path/'status.json',state);write(path/'screen_complete.json',dict(complete=stop==100,epoch_completed=stop,latest_checkpoint=str(cp/'latest.pth'),best_epoch=best_epoch,best_loss=best_loss,test_data_accessed=False));print('PAUSED_NO_AUTO_CONTINUATION',arm,stop,flush=True)

def main():
 mode=sys.argv[1];arm='DualA';assert mode in ('preflight','train');assert '--stop300' not in sys.argv;m=verify();path=RUN/arm/mode;path.mkdir(parents=True,exist_ok=True)
 assert json.loads((RUN/'cpu_acceptance.json').read_text())['passed'];assert torch.cuda.is_available(),'no CPU fallback'
 api=runner.baseline_api();api.seed_all(42);torch.set_num_threads(4);torch.cuda.set_device(0);device=torch.device('cuda:0')
 model=build_candidate(True).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8)
 # EXACT original A fresh300 timeline: periodic cosine T_max50, not a new 100-epoch schedule.
 scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=m['config']['training']['T_max_epochs'],eta_min=1e-5)
 loaders,generator=make_fast_loaders(m['config'],api,workers=2,validation_batch=8);criterion=api.BceDiceLoss().to(device)
 (RUN/arm/(mode+'.pid')).write_text(str(os.getpid())+'\n')
 mapping=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
 assert any(UUID in l and f', {os.getpid()},' in l for l in mapping.splitlines()),mapping
 assert sum(p.numel() for p in model.parameters())==json.loads((RUN/'cpu_acceptance.json').read_text())['parameters']
 write(path/'provenance.json',dict(arm=arm,identity=ARMS[arm],pid=os.getpid(),physical_gpu=1,gpu_uuid=UUID,parameters=sum(p.numel() for p in model.parameters()),fresh_start='--resume' not in sys.argv,imagenet_only=True,protocol_sha256=sha256_file(RUN/'protocol.json'),test_data_accessed=False));write(path/'weight_loading_audit.json',model.encoder.loading_audit)
 boundary={}
 def stage_hook(mod,ins,out):
  if not boundary:
   assert list(ins[0].shape)==[8,512,16,16] and list(out.shape)==[8,256,16,16]
   boundary.update(input=list(ins[0].shape),output=list(out.shape));write(path/'real_stage_boundary.json',boundary)
 model.decoder.stages[2].register_forward_hook(stage_hook)
 print('ENTRY_IDENTITY',arm,mode,'PID',os.getpid(),'UUID',UUID,'PATH',path,flush=True)
 try:
  if mode=='preflight':
   write(path/'ready.json',dict(pid=os.getpid(),ready=True));preflight(arm,model,opt,scheduler,loaders,generator,criterion,device,path)
  elif mode=='train':
   assert (RUN/'PREFLIGHT_ALL_PASSED').exists();write(path/'ready.json',dict(pid=os.getpid(),ready=True));wait_marker('TRAIN_GO');train(arm,model,opt,scheduler,loaders,generator,criterion,device,path,resume='--resume' in sys.argv,stop=100)
  else:raise ValueError(mode)
 finally:
  for l in loaders.values():
   if hasattr(l,'close'):l.close()
if __name__=='__main__':
 try:main()
 except BaseException as e:
  write(RUN/'DualA'/'failure.json',dict(error=repr(e),pid=os.getpid()));raise

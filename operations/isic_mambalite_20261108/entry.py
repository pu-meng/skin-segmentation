"""Public MambaLiteUNet seed42 on the same deduplicated ISIC2018 split."""
import os,sys,json,copy,io,shutil,subprocess,fcntl,traceback,time,types
from pathlib import Path
import numpy as np
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation');OPS=Path(__file__).resolve().parent
RUN=Path('/media/8T/PuMengYu/experiments/skin_segmentation/ISIC2018_MambaLite_public_FRESH300_20261108');ARM='MambaLite';GPU=1
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
from operations.isic_a01_20261108 import entry as common
write=common.write;sha=common.sha;digest=common.digest
REF=Path('/media/8T/PuMengYu/experiments/skin_segmentation/ISIC2018_A0_A1_FRESH300_20261108/protocol.json')
BASE=ROOT/'source/config/protocol_v5_dedup_seed42.json'
def imports():
 import ham_fast_runner as runner
 api=runner.baseline_api()
 import torch
 from source.isic2018 import data
 return torch,runner,api,data

def predictor(torch):
 class Predictor(torch.nn.Module):
  def __init__(self,model,tta):
   super().__init__();self.model=model;self.tta=tta;self.raw=None;self.times=[];self.verify_remaining=4
   self.handle=model.final.register_forward_hook(self.capture)
  def capture(self,module,inputs,output):self.raw=output
  def logits(self,x):
   self.raw=None;prob=self.model(x);assert self.raw is not None
   logit=torch.nn.functional.interpolate(self.raw,scale_factor=2,mode='bilinear',align_corners=True)
   if self.verify_remaining:
    torch.testing.assert_close(logit.sigmoid(),prob,rtol=0,atol=0);self.verify_remaining-=1
   return logit
  def forward(self,x):
   torch.cuda.synchronize();start=time.perf_counter();z=self.logits(x)
   if self.tta:
    for axes in [(2,),(3,),(2,3)]:z=z+torch.flip(self.logits(torch.flip(x,axes)),axes)
    z=z/4
   result=z.sigmoid();assert torch.isfinite(result).all();torch.cuda.synchronize();self.times.append(time.perf_counter()-start);return result
  def close(self):self.handle.remove()
 return Predictor

def prepare():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not RUN.exists()
 torch,runner,api,data=imports();torch.set_num_threads(1);meta=data.verify();parent=common.read(REF);cfg=copy.deepcopy(parent['config'])
 assert meta['split_sha256']==cfg['data']['split_sha256'] and meta['counts']==dict(train=1780,val=258,test=518)
 audit=common.read(data.DEDUP/'split_audit.json');assert audit['checks']['unresolved_review_pairs']==audit['checks']['remaining_multi_active_original_strong_components']==0
 assert sha(data.DEDUP/'split.csv')==audit['split_sha256']==cfg['data']['dedup_split_sha256']
 allowed=common.read(BASE)['models']['mambalite']['inherited_unused_parameters']
 api.seed_all(42);m=api.MambaLiteUNet();assert sum(p.numel() for p in m.parameters())==910627
 buf=io.BytesIO();torch.save(m.state_dict(),buf);buf.seek(0);m.load_state_dict(torch.load(buf,map_location='cpu',weights_only=False),strict=True);initial=digest(m.state_dict());assert not torch.cuda.is_initialized()
 sources={str(REF):sha(REF),str(BASE):sha(BASE)}
 for mod in list(sys.modules.values()):
  f=getattr(mod,'__file__',None)
  if f and str(f).startswith(str(ROOT)) and f.endswith('.py') and Path(f).exists():sources[str(Path(f).resolve())]=sha(f)
 for f in [Path(__file__),ROOT/'source/bottleneck_mamba/final_test_pipeline.py',ROOT/'operations/nnunet_mirror_validation/run_validation.py',ROOT/'operations/nnunet_mirror_validation/mirror_core.py']:sources[str(f)]=sha(f)
 RUN.mkdir();write(RUN/'protocol.json',dict(identity='author public MambaLiteUNet four channel branches ISIC2018 fresh300 seed42',config=cfg,initial_state_sha256=initial,parameters=910627,allowed_missing_gradients=allowed,protected_sha256=sources,initialization='official public defaults, random seed42; no ImageNet or HAM segmentation weights',same_data_and_training_as=str(REF),contract='300epochs66900updates batch8 FP32 AdamW lr.001 original cosineTmax50; earliest minimum validation loss; fixedbest518 none+mirror4; no test selection',scheduling='explicit user authorization immediate shared GPU1; preflight3 true optimizer steps gates fresh training; no other tasks modified',limitations='public code local reproduction, not Figure2b reconstruction; single seed; ISIC2018 custom deduplicated split, not an author benchmark identity claim'))
 for f,h in sources.items():
  dst=RUN/'source_snapshot'/Path(f).relative_to('/');dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dst);assert sha(dst)==h
 write(RUN/ARM/'cpu_acceptance.json',dict(passed=True,parameters=910627,random_initialization=True,full_state_strict_roundtrip=True,initial_state_sha256=initial,cuda_initialized=False,cache_hashes_verified=True,split_sha256=meta['split_sha256']))
 print('CPU_ACCEPTED_PUBLIC_MAMBALITE',910627,flush=True)
def verify():
 p=common.read(RUN/'protocol.json')
 for f,h in p['protected_sha256'].items():assert sha(f)==h,'source drift '+f
 assert sha(p['config']['data']['split_file'])==p['config']['data']['split_sha256'];assert shutil.disk_usage(RUN).free>8*2**30
 return p

def phase(mode):
 p=verify();torch,runner,api,data=imports();torch.set_num_threads(4);api.seed_all(42);m=api.MambaLiteUNet();work=RUN/ARM
 assert digest(m.state_dict())==p['initial_state_sha256'] and not torch.cuda.is_initialized()
 if mode=='test':
  common.RUN=RUN;common.verify=verify
  import operations.nnunet_mirror_validation.run_validation as test_runtime
  test_runtime.Predictor=predictor(torch)
  return common.test(ARM,GPU,p,m,torch,runner,data)
 stage=work/('preflight' if mode=='preflight' else 'run');stage.mkdir()
 if mode=='train':assert common.read(work/'preflight/acceptance.json')['status']=='passed'
 torch.cuda.set_device(0);m.cuda();loaders,g=data.loaders(p['config'],api);crit=api.BceDiceLoss().cuda();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8)
 info=dict(model='MambaLiteUNet public ISIC2018',dataset='ISIC2018',parameters=910627,pid=os.getpid(),seed=42,physical_gpu=1,gpu_uuid=common.UUIDS[1],protocol_sha256=sha(RUN/'protocol.json'),split_sha256=p['config']['data']['split_sha256'],training=p['config']['training'],source_sha256=p['protected_sha256'],fresh_start=True,test_data_accessed=False)
 write(stage/'provenance.json',info);write(stage/'weight_loading_audit.json',dict(initialization='random seed42; no checkpoint loaded'))
 try:
  if mode=='preflight':
   q=runner.preflight(m,loaders,crit,opt,torch.device('cuda:0'),stage,p['allowed_missing_gradients'])
   m.eval();x=next(iter(loaders['val_precise']))[0].cuda()
   with torch.no_grad():
    for tta in [False,True]:
     wrapper=predictor(torch)(m,tta);y=wrapper(x);assert y.shape==(1,1,256,256) and torch.isfinite(y).all();wrapper.close()
   q.update(mambalite_final_logits_roundtrip=True,both_inference_modes_checked_on_validation=True);q.update(protocol_sha256=info['protocol_sha256'],physical_gpu=1,gpu_uuid=common.UUIDS[1]);write(stage/'acceptance.json',q);print('REAL_BATCH8_PREFLIGHT_PASS',flush=True)
  else:
   original=opt.step;first=True
   def step(self,*a,**kw):
    nonlocal first
    if first:
     missing={n for n,q in m.named_parameters() if q.requires_grad and q.grad is None};assert missing==set(p['allowed_missing_gradients']);assert bool(torch.stack([torch.isfinite(q.grad).all() for q in m.parameters() if q.grad is not None]).all())
     name,q=next((n,q) for n,q in m.named_parameters() if q.grad is not None and bool(q.grad.abs().max()>0));before=q.detach().clone()
    out=original(*a,**kw)
    if first:
     delta=float((q-before).abs().max());assert delta>0;write(work/'first_optimizer_update.json',dict(pid=os.getpid(),physical_gpu=1,gpu_uuid=common.UUIDS[1],batch=8,finite_gradients=True,probe=name,max_delta=delta));print('FIRST_FORMAL_UPDATE',flush=True);first=False
    return out
   opt.step=types.MethodType(step,opt);runner.train(m,loaders,crit,opt,torch.device('cuda:0'),stage,p['config'],info,g,resume=False)
   audit=common.read(stage/'artifact_audit.json');assert audit['training_complete'] and audit['epochs']==300 and audit['optimizer_updates']==66900
   f=stage/'validation/summary.json';s=common.read(f);s['dataset']='ISIC2018';write(f,s);(stage/'validation/report.txt').write_text(f'ISIC2018 public MambaLiteUNet seed42,300epochs; selected epoch {s["selected_epoch"]}; validation258.\n')
 finally:
  for l in loaders.values():
   if hasattr(l,'close'):l.close()
   elif getattr(l,'_iterator',None) is not None:l._iterator._shutdown_workers()
def controller():
 work=RUN/ARM
 with (work/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for mode in ['preflight','train','test']:
   verify();env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=common.UUIDS[1],EXPERIMENT_PHYSICAL_GPU='1',EXPERIMENT_GPU_UUID=common.UUIDS[1],PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_THREADING_LAYER='GNU')
   with (work/(mode+'.log')).open('x') as log:
    child=subprocess.Popen([common.PY,'-B','-u',__file__,mode],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);write(work/'controller_status.json',dict(status=mode,controller_pid=os.getpid(),child_pid=child.pid,physical_gpu=1,updated_unix=time.time()));rc=child.wait()
   write(work/(mode+'_exitcode.json'),dict(exitcode=rc,pid=child.pid));assert rc==0,mode+' failed; downstream blocked'
  assert common.read(work/'completion.json')['complete'];write(work/'controller_status.json',dict(status='complete',physical_gpu=1,controller_pid=os.getpid()))
if __name__=='__main__':
 try:
  if sys.argv[1]=='prepare':prepare()
  elif sys.argv[1]=='controller':controller()
  else:phase(sys.argv[1])
 except BaseException as e:
  if RUN.exists():write(RUN/ARM/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),pid=os.getpid()))
  raise

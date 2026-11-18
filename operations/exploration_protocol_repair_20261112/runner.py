import os,sys,json,time,fcntl,subprocess,traceback,hashlib,shutil,types,importlib
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
OUT=Path('/media/8T/PuMengYu/experiments/skin_segmentation/EXPLORATION_REPAIR_FRESH300_20261112')
PY='/home/PuMengYu/anaconda3/envs/medseg/bin/python'
GPU='GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b'
ARMS=[('keepS2','ISIC2018'),('keepS3','ISIC2018'),('convnextS','ISIC2018'),('D2Mskips','ISIC2018'),('MSDdec','ISIC2018'),('swiftS2S3','HAM10000')]
MOD={'keepS2':('source.a1_s2s3_20261111.candidate',{'use_skip2':True,'use_skip3':False}),'keepS3':('source.a1_s2s3_20261111.candidate',{'use_skip2':False,'use_skip3':True}),'convnextS':('source.a1_s2s3_convnext_20261111.candidate',{'use_skip2':True,'use_skip3':True}),'D2Mskips':('source.a1_swift_d2m_20261111.candidate',{'variant':'d2m_skips'}),'MSDdec':('source.a1_swift_msddec_20261111.candidate',{'variant':'msddec'}),'swiftS2S3':('source.swift_a1_s2s3_20261111.candidate',{'use_skip2':True,'use_skip3':True})}
REF={'ISIC2018':Path('/media/8T/PuMengYu/experiments/skin_segmentation/ISIC2018_A0_A1_FRESH300_20261108/protocol.json'),'HAM10000':Path('/media/8T/PuMengYu/experiments/skin_segmentation/DecoderDualA_RESUME100to300_20261106/protocol.json')}
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba'),str(ROOT/'source/three_module_impl_20261104'),str(ROOT/'source/isic2018')]
def write(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str)+'\n');t.replace(p)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def setup():
 import torch,ham_fast_runner as runner
 from pretrained_unet2d import prepare_imports
 prepare_imports('mambavision_t');prepare_imports('tinyvim_s')
 return torch,runner,runner.baseline_api()
def build(a,pre):
 module,kw=MOD[a];return importlib.import_module(module).build_candidate(pretrained=pre,**kw)
def config(ds):return read(REF[ds])['config']
def loaders(ds,c,api):
 if ds=='ISIC2018':
  from source.isic2018 import data
  return data.loaders(c,api)
 from ham_fast_pipeline import make_fast_loaders
 return make_fast_loaders(c,api,workers=2,validation_batch=8)
def close(ls):
 for l in ls.values():
  if hasattr(l,'close'):l.close()
  elif getattr(l,'_iterator',None) is not None:l._iterator._shutdown_workers()
def phase(a,ds,kind):
 torch,runner,api=setup();torch.set_num_threads(4);api.seed_all(42);c=config(ds);w=OUT/a;w.mkdir(exist_ok=True)
 assert c['training']['epochs']==300 and c['training']['batch_size']==8
 assert sha(c['data']['split_file'])==c['data']['split_sha256']
 m=build(a,kind!='test');d=m.decoder
 if a in ('keepS2','keepS3','convnextS','swiftS2S3'):
  assert (d.skip2 is not None)==(a!='keepS3') and (d.skip3 is not None)==(a!='keepS2')
 if a=='D2Mskips':assert type(d).__name__=='DecoderD2M'
 if a=='MSDdec':assert type(d).__name__=='DecoderMSD'
 if kind=='cpu':
  import io
  buf=io.BytesIO();torch.save(m.state_dict(),buf);buf.seek(0);m.load_state_dict(torch.load(buf,map_location='cpu',weights_only=False),strict=True)
  ls,g=loaders(ds,c,api);counts={k:len(ls[k].dataset) for k in ('train','val')};steps=len(ls['train']);close(ls)
  expected={'ISIC2018':({'train':1780,'val':258},223),'HAM10000':({'train':6983,'val':1010},873)}
  assert (counts,steps)==expected[ds],(counts,steps)
  assert not torch.cuda.is_initialized()
  sources={}
  for mod in list(sys.modules.values()):
   f=getattr(mod,'__file__',None)
   if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py') and Path(f).is_file():sources[str(Path(f).resolve())]=sha(f)
  sources[str(Path(__file__))]=sha(__file__);sources[str(REF[ds])]=sha(REF[ds]);sources[c['data']['split_file']]=sha(c['data']['split_file'])
  write(w/'cpu_acceptance.json',{'passed':True,'arm':a,'dataset':ds,'parameters':sum(p.numel() for p in m.parameters()),'decoder':type(d).__name__,'strict_serialized_roundtrip':True,'counts':counts,'steps':steps,'cuda_initialized':False,'sources':sources})
  print('CPU_ACCEPTED',a,counts,steps,flush=True);return
 proof=read(w/'cpu_acceptance.json')
 for f,h in proof['sources'].items():assert sha(f)==h,'source drift: '+f
 assert shutil.disk_usage(OUT).free>8*2**30
 device=torch.device('cuda:0')
 if kind=='test':
  from final_test_pipeline import fixed_best_record,validate_outputs,training_ready
  from operations.nnunet_mirror_validation.run_validation import Predictor
  from torch.utils.data import DataLoader
  assert training_ready(w/'run');fixed=fixed_best_record(w/'run');assert fixed['history_rows']==300
  target=w/'final_test';target.mkdir(exist_ok=False);cp=target/'fixed_best.pth';shutil.copy2(fixed['checkpoint'],cp);assert sha(cp)==fixed['checkpoint_sha256']
  m.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False),strict=True)
  write(target/'cpu_load_audit.json',{'strict':True,'checkpoint_sha256':sha(cp),'selected_epoch':fixed['selected_epoch'],'cuda_initialized':torch.cuda.is_initialized()})
  if ds=='ISIC2018':
   from source.isic2018 import data
   dataset=data.ISICDataset(data.CACHE,'test');expected=518
  else:
   from project_paths import PATHS
   dataset=api.HAMDataset(Path(PATHS['active_split']).parent,'test');expected=2020
  ids={r['image_id'] for r in dataset.rows};assert len(dataset)==len(ids)==expected and not dataset.training
  loader=DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0);m=m.to(device).eval()
  for mode in ('none','mirror4'):
   pred=Predictor(m,mode=='mirror4').eval();dest=target/mode;assert not dest.exists()
   result=api.evaluate(pred,loader,api.BceDiceLoss().to(device),device,dest)
   assert len(pred.times)==expected and pred.verify_remaining==0
   result.update(dataset=ds,model=a,split='test',checkpoint_sha256=sha(cp),selected_epoch=fixed['selected_epoch'],split_sha256=c['data']['split_sha256'],tta=mode=='mirror4',training_epochs=300,physical_gpu=1)
   audit=validate_outputs(dest,ids,result);assert audit['complete'];write(dest/'summary.json',result);write(dest/'artifact_audit.json',audit);pred.close();print('TEST_COMPLETE',a,mode,flush=True)
  write(w/'completion.json',{'complete':True,'epoch_budget':300,'selected_epoch':fixed['selected_epoch'],'checkpoint_sha256':sha(cp),'test_images_each_mode':expected});return
 ls,g=loaders(ds,c,api);m=m.to(device);criterion=api.BceDiceLoss().to(device);t=c['training'];opt=torch.optim.AdamW(m.parameters(),lr=t['lr'],weight_decay=t['weight_decay'],betas=tuple(t['betas']),eps=t['eps'])
 try:
  if kind=='preflight':
   m.train();x,y,_,_=next(iter(ls['train']));assert tuple(x.shape)==(8,3,256,256);opt.zero_grad(set_to_none=True);pred=m(x.to(device));assert pred.min()>=0 and pred.max()<=1
   loss=criterion(pred,y.to(device));assert torch.isfinite(loss);loss.backward();gs=[p.grad for p in m.parameters() if p.grad is not None];assert gs and all(bool(torch.isfinite(z).all()) for z in gs);opt.step();torch.cuda.synchronize()
   write(w/'preflight.json',{'passed':True,'loss':float(loss),'batch_shape':list(x.shape),'peak_reserved':torch.cuda.max_memory_reserved(),'pid':os.getpid(),'gpu_uuid':GPU,'probability_range':[float(pred.min()),float(pred.max())]});print('PREFLIGHT_PASS',a,float(loss),flush=True);return
  assert read(w/'preflight.json')['passed'];stage=w/'run';stage.mkdir(exist_ok=False)
  prov={'model':a,'dataset':ds,'seed':42,'split_sha256':c['data']['split_sha256'],'parameters':sum(p.numel() for p in m.parameters()),'protocol_sha256':sha(OUT/'protocol.json'),'engineering':{'runner':'ham_fast_runner.train','workers':2,'validation_batch':8}}
  write(stage/'provenance.json',prov);original=opt.step;first=True
  def observed(self,*args,**kw):
   nonlocal first
   if first:
    name,p=next((n,p) for n,p in m.named_parameters() if p.grad is not None and bool(torch.isfinite(p.grad).all()) and bool((p.grad!=0).any()));before=p.detach().clone()
   r=original(*args,**kw)
   if first:
    torch.cuda.synchronize();delta=float((p-before).abs().max());assert delta>0
    write(w/'first_optimizer_update.json',{'pid':os.getpid(),'physical_gpu':1,'gpu_uuid':GPU,'parameter':name,'max_delta':delta,'epoch':1,'step':1});print('FIRST_FORMAL_OPTIMIZER_UPDATE',a,name,delta,flush=True);first=False
   return r
  opt.step=types.MethodType(observed,opt)
  runner.train(m,ls,criterion,opt,device,stage,c,prov,g,resume=False)
  from final_test_pipeline import training_ready
  assert training_ready(stage);print('TRAIN_COMPLETE',a,flush=True)
 finally:close(ls)
def controller():
 OUT.mkdir(parents=True,exist_ok=True)
 with (OUT/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  deadline=time.time()+30*86400
  proto={'arms':[{'arm':a,'dataset':d} for a,d in ARMS],'physical_gpu':1,'gpu_uuid':GPU,'deadline_unix':deadline,'epochs':300,'scheduler':'original candidate ham_fast_runner CosineAnnealingLR T_max=50, eta_min=1e-5; differs from erroneous exploratory T_max300','selection':'earliest strict minimum validation loss with original batch1 potential-best recheck','outputs':'probability; BceDiceLoss; no extra sigmoid; test logits head hook verified against forward','augmentation':'original dataset-specific planned samplers','shared_gpu':True,'ref_sha256':{k:sha(v) for k,v in REF.items()},'entry_sha256':sha(__file__)}
  p=OUT/'protocol.json';assert not p.exists();write(p,proto)
  state={'status':'running','controller_pid':os.getpid(),'deadline_unix':deadline,'arms':[{**x,'status':'queued'} for x in proto['arms']]};write(OUT/'queue_status.json',state)
  for idx,(a,ds) in enumerate(ARMS):
   w=OUT/a;w.mkdir(exist_ok=True)
   for kind in ('cpu','preflight','train','test'):
    attempt=0
    while True:
     assert time.time()<deadline,'total deadline exceeded'
     if kind in ('preflight','train','test'):
      while True:
       assert time.time()<deadline,'GPU wait deadline exceeded'
       raw=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True)
       free=int(next(line.split(',')[1].strip() for line in raw.splitlines() if line.split(',')[0].strip()==GPU))
       state.update(status='waiting_gpu' if free<6000 else 'running',arm=a,phase=kind,free_mib=free,minimum_free_mib=6000,updated_unix=time.time());state['arms'][idx]['status']=state['status']+'_'+kind;write(OUT/'queue_status.json',state)
       if free>=6000:break
       time.sleep(60)
     attempt+=1;env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='' if kind=='cpu' else GPU,EXPERIMENT_PHYSICAL_GPU='1',EXPERIMENT_GPU_UUID=GPU,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
     log=w/(kind+('_attempt%d'%attempt if attempt>1 else '')+'.log')
     with log.open('x') as f:
      child=subprocess.Popen([PY,'-B','-u',__file__,'phase',a,ds,kind],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
      state.update(status='running',arm=a,phase=kind,child_pid=child.pid,updated_unix=time.time());state['arms'][idx]['status']=kind;write(OUT/'queue_status.json',state)
      try:rc=child.wait(timeout=max(1,deadline-time.time()))
      except subprocess.TimeoutExpired:
       child.terminate();child.wait(timeout=60);raise RuntimeError('total deadline exceeded')
     write(w/(kind+('_attempt%d'%attempt if attempt>1 else '')+'_exit.json'),{'exitcode':rc,'pid':child.pid,'time':time.time()})
     if rc==0:break
     text=log.read_text(errors='replace')
     if kind=='preflight' and ('OutOfMemoryError' in text or 'CUDA out of memory' in text):
      state.update(status='waiting_gpu_after_oom',reason='formal batch preflight OOM; wait and retry without changing batch',updated_unix=time.time());write(OUT/'queue_status.json',state);time.sleep(120);continue
     raise RuntimeError('%s %s failed exit %s; see %s'%(a,kind,rc,log))
   state['arms'][idx]['status']='complete';write(OUT/'queue_status.json',state)
  state.update(status='complete',updated_unix=time.time());write(OUT/'queue_status.json',state)
if __name__=='__main__':
 try:
  if sys.argv[1]=='controller':controller()
  elif sys.argv[1]=='phase':phase(*sys.argv[2:])
 except BaseException as e:
  if OUT.exists():
   p=OUT/'queue_status.json';s=read(p) if p.exists() else {};s.update(status='failed',error=repr(e),traceback=traceback.format_exc(),updated_unix=time.time());write(p,s)
  raise

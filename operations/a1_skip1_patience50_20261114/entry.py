"""A1 SwiftFormer-S seed42 repeats with validation-loss patience 50."""
import os,sys,json,copy,csv,hashlib,shutil,subprocess,time,fcntl,traceback,io,types
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
OPS=Path(__file__).resolve().parent
RUN=Path(os.environ['A1_PATIENCE_RUN_DIR'])
PY='/home/PuMengYu/anaconda3/envs/medseg/bin/python'
UUIDS=['GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde','GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b']
ARMS=['A1-skip1-patience50']
REF=ROOT/'log/TinyViM_四层_E0_E1E3_ISIC2018_20261105/protocol.json'
sys.path[:0]=[str(ROOT/'operations/a1_seed42_patience50_20261114'),str(OPS),str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
def read(p):return json.loads(Path(p).read_text())
def write(p,o):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(o,ensure_ascii=False,indent=2));t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def digest(sd):
 h=hashlib.sha256()
 for n,t in sorted(sd.items()):h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def imports(arm):
 from pretrained_unet2d import prepare_imports
 prepare_imports('mambavision_t');prepare_imports('tinyvim_s')
 import torch,ham_fast_runner as runner
 from source.isic2018 import data
 from source.a1_swift_skip_position_20261114.candidate import build_candidate as build_skip
 build_candidate=lambda pretrained: build_skip(1,pretrained)
 return torch,runner,data,build_candidate

def prepare():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not RUN.exists()
 torch,runner,data,_=imports('A1');torch.set_num_threads(1)
 meta=data.verify();audit=read(data.DEDUP/'split_audit.json');assert audit['checks']['unresolved_review_pairs']==audit['checks']['remaining_multi_active_original_strong_components']==0
 assert sha(data.DEDUP/'split.csv')==audit['split_sha256']
 for name,key in [('excluded_images.csv','excluded_csv_sha256'),('source_inventory.csv','source_inventory_sha256')]:assert sha(data.DEDUP/name)==audit[key]
 assert sha(read(data.DEDUP/'exclusion_policy.json')['evidence'])==audit['decision_sha256']
 cfg={k:copy.deepcopy(read(REF)[k]) for k in ['data','training','preprocessing','evaluation']}
 assert meta['counts']==cfg['data']['counts']==dict(train=1780,val=258,test=518)
 assert sha(data.CACHE/'cache_complete.json')==cfg['data']['cache_manifest_sha256']
 assert cfg['training']['epochs']==300 and cfg['training']['total_optimizer_updates']==66900
 cfg['training']['early_stopping']=True
 cfg['training']['early_stopping_patience']=50
 cfg['training']['early_stopping_monitor']='validation loss; strict improvement resets patience'
 cfg['evaluation']['completion']='up to 300 epochs; stop after 50 consecutive non-improving validation epochs; val-minloss fixed best -> CPU strict load -> test518 none and mirror4 immediately after training'
 RUN.mkdir();records={}
 api=runner.baseline_api()
 for arm in ARMS:
  _,_,_,build=imports(arm);api.seed_all(42);m=build(True);assert all(p.requires_grad for p in m.parameters())
  state=digest(m.state_dict());buf=io.BytesIO();torch.save(m.state_dict(),buf);buf.seek(0);m.load_state_dict(torch.load(buf,map_location='cpu',weights_only=False),strict=True)
  records[arm]=dict(initial_state_sha256=state,parameters=sum(p.numel() for p in m.parameters()),state_bytes=sum(t.numel()*t.element_size() for t in m.state_dict().values()),encoder_loading_audit=m.encoder.loading_audit,model_class=type(m).__name__,D2_class=type(m.decoder.stages[2]).__name__)
  assert not torch.cuda.is_initialized();write(RUN/arm/'cpu_acceptance.json',dict(passed=True,full_state_strict_roundtrip=True,cuda_initialized=False,**records[arm]));print('CPU_ACCEPTED',arm,records[arm]['parameters'],flush=True);del m,buf
 sources={str(REF):sha(REF)}
 for mod in list(sys.modules.values()):
  f=getattr(mod,'__file__',None)
  if f and str(f).startswith(str(ROOT)) and f.endswith('.py') and Path(f).exists():sources[str(Path(f).resolve())]=sha(f)
 for f in [Path(__file__),ROOT/'source/bottleneck_mamba/final_test_pipeline.py',ROOT/'operations/nnunet_mirror_validation/run_validation.py',ROOT/'operations/nnunet_mirror_validation/mirror_core.py'] :sources[str(f)]=sha(f)
 # Include the exact public initial classifier weight, rather than any segmentation best.
 from project_paths import CONFIGS
 for arm in ARMS:
  for key,value in records[arm]['encoder_loading_audit'].items():
   if isinstance(value,str) and Path(value).is_file():sources[value]=sha(value)
 needed=sum(r['state_bytes'] for r in records.values())*12+5*2**30
 assert shutil.disk_usage(RUN).free>needed+8*2**30
 p=dict(identity='A1 skip1 ISIC2018 seed42 fresh up-to-300 with validation patience50',config=cfg,arms=records,protected_sha256=sources,initialization='Original skip1 architecture and seed42 initialization; official ImageNet encoder only; no HAM segmentation weights',data_audit=audit,contract='ISIC2018 dedup train1780 val258 test518; seed42 fresh start; max300 epochs batch8 FP32 AdamW lr.001 cosineTmax50; stop after 50 consecutive validation-loss non-improvements; retain lowest-validation-loss checkpoint; test518 none+mirror4 starts immediately after training without a GPU-free-memory gate',scheduling='Independent A1 seed42 repeat; 4GB training/preflight gate with existing compute-process cap; final test bypasses free-memory admission after training',storage_required_bytes=needed)
 write(RUN/'protocol.json',p)
 for f,h in sources.items():
  dst=RUN/'source_snapshot'/Path(f).relative_to('/');dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dst);assert sha(dst)==h
 write(RUN/'prepare_complete.json',dict(passed=True,arms=ARMS,dedup_cache_hashes_verified=True,protocol_sha256=sha(RUN/'protocol.json'),early_stopping_patience=50,max_epochs=300))

def verify():
 p=read(RUN/'protocol.json')
 for f,h in p['protected_sha256'].items():assert sha(f)==h,'source drift '+f
 assert sha(p['config']['data']['split_file'])==p['config']['data']['split_sha256']
 assert shutil.disk_usage(RUN).free>8*2**30
 return p

def phase(arm,mode,gpu):
 p=verify();cfg=p['config'];work=RUN/arm;torch,runner,data,build=imports(arm);torch.set_num_threads(4);api=runner.baseline_api();api.seed_all(42);m=build(mode!='test')
 if mode=='test':return test(arm,gpu,p,m,torch,runner,data)
 assert digest(m.state_dict())==p['arms'][arm]['initial_state_sha256'] and not torch.cuda.is_initialized()
 stage=work/('preflight' if mode=='preflight' else 'run');stage.mkdir()
 if mode=='train':assert read(work/'preflight/acceptance.json')['status']=='passed'
 torch.cuda.set_device(0);m.cuda();loaders,g=data.loaders(cfg,api);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8);crit=api.BceDiceLoss().cuda()
 info=dict(model=arm+' ISIC2018',dataset='ISIC2018',pid=os.getpid(),seed=42,gpu_uuid=UUIDS[gpu],physical_gpu=gpu,protocol_sha256=sha(RUN/'protocol.json'),split_sha256=cfg['data']['split_sha256'],source_sha256=p['protected_sha256'],fresh_start=True,training=cfg['training'],test_data_accessed=False)
 write(stage/'provenance.json',info);write(stage/'weight_loading_audit.json',m.encoder.loading_audit)
 try:
  if mode=='preflight':
   before={n:q.detach().clone() for n,q in m.decoder.stages[2].named_parameters()}
   proof=runner.preflight(m,loaders,crit,opt,torch.device('cuda:0'),stage,[])
   changed=[n for n,q in m.decoder.stages[2].named_parameters() if not torch.equal(before[n],q.detach())];assert changed
   proof.update(D2_changed_tensors=changed,protocol_sha256=info['protocol_sha256'],split_sha256=info['split_sha256'],physical_gpu=gpu,gpu_uuid=UUIDS[gpu]);write(stage/'acceptance.json',proof);print('REAL_BATCH8_PREFLIGHT_PASS',arm,flush=True)
  else:
   original=opt.step;first=True
   def step(self,*args,**kwargs):
    nonlocal first
    if first:
     grads=[q.grad for q in m.parameters() if q.requires_grad];assert all(v is not None for v in grads) and bool(torch.stack([torch.isfinite(v).all() for v in grads]).all())
     name,q=next((n,q) for n,q in m.named_parameters() if n.startswith('decoder.stages.2.') and bool(q.grad.abs().max()>0));before=q.detach().clone()
    result=original(*args,**kwargs)
    if first:
     delta=float((q-before).abs().max());assert delta>0;write(work/'first_optimizer_update.json',dict(pid=os.getpid(),physical_gpu=gpu,gpu_uuid=UUIDS[gpu],batch=8,input=[8,3,256,256],finite_gradients=True,probe=name,max_delta=delta,protocol_sha256=info['protocol_sha256']));print('FIRST_FORMAL_UPDATE',arm,flush=True);first=False
    return result
   opt.step=types.MethodType(step,opt);runner.train(m,loaders,crit,opt,torch.device('cuda:0'),stage,cfg,info,g,resume=False)
   a=read(stage/'artifact_audit.json');assert a['training_complete'] and 1<=a['epochs']<=300 and a['optimizer_updates']==a['epochs']*cfg['training']['updates_per_epoch'] and a['early_stopping_patience']==50
   f=stage/'validation/summary.json';s=read(f);s.update(dataset='ISIC2018');write(f,s);(stage/'validation/report.txt').write_text(f'ISIC2018 {arm}: {a["epochs"]}/300 epochs (patience=50), validation258, best epoch {s["selected_epoch"]}; see summary.json.\n')
 finally:
  for l in loaders.values():
   if hasattr(l,'close'):l.close()
   elif getattr(l,'_iterator',None) is not None:l._iterator._shutdown_workers()

def test(arm,gpu,p,m,torch,runner,data):
 from torch.utils.data import DataLoader
 from final_test_pipeline import fixed_best_record,validate_outputs
 from operations.nnunet_mirror_validation.run_validation import Predictor
 work=RUN/arm;fixed=fixed_best_record(work/'run');out=work/'final_test';out.mkdir();snap=out/'fixed_best.pth';shutil.copy2(fixed['checkpoint'],snap);assert sha(snap)==fixed['checkpoint_sha256']
 m.load_state_dict(torch.load(snap,map_location='cpu',weights_only=False),strict=True);assert not torch.cuda.is_initialized();write(out/'cpu_load_audit.json',dict(strict=True,**fixed))
 api=runner.baseline_api();api.seed_all(42);ds=data.ISICDataset(data.CACHE,'test');ids={r['image_id'] for r in ds.rows};assert len(ds)==len(ids)==518 and not ds.training
 loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=0);torch.cuda.set_device(0);m.cuda().eval();results={}
 for mode in ['none','mirror4']:
  target=out/mode;pred=Predictor(m,mode=='mirror4').eval();result=api.evaluate(pred,loader,api.BceDiceLoss().cuda(),torch.device('cuda:0'),target);assert len(pred.times)==518 and pred.verify_remaining==0
  training_epochs=read(work/'run/artifact_audit.json')['epochs']
  result.update(dataset='ISIC2018',model=arm,selected_epoch=fixed['selected_epoch'],checkpoint_sha256=sha(snap),checkpoint=str(snap),split='test',seed=42,tta=mode=='mirror4',training_epochs=training_epochs,split_sha256=p['config']['data']['split_sha256'],test_used_for_checkpoint_selection=False)
  audit=validate_outputs(target,ids,result);assert audit['complete'];write(target/'summary.json',result);write(target/'artifact_audit.json',audit);pred.close();results[mode]=result
 verify();artifact=read(work/'run/artifact_audit.json');write(work/'completion.json',dict(complete=True,epochs=artifact['epochs'],max_epochs=300,early_stopping_patience=50,test_images_per_mode=518,results=results))

def controller(arm,gpu):
 work=RUN/arm
 with (work/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for mode in ['preflight','train','test']:
   verify();env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=UUIDS[gpu],EXPERIMENT_PHYSICAL_GPU=str(gpu),EXPERIMENT_GPU_UUID=UUIDS[gpu],OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
   admission=None
   if mode!='test':
    gate=Path('/home/PuMengYu/experiments/skin_segmentation/operations/encoder_swap_3slot_takeover_20261108/admission.lock')
    while True:
     admission=gate.open('a');fcntl.flock(admission,fcntl.LOCK_EX)
     lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines()
     pids=[int(x.split(',')[1]) for x in lines if x.startswith(UUIDS[gpu])]
     stats=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader'],text=True).splitlines()
     free=int(next(x.split(',')[1].strip().split()[0] for x in stats if x.startswith(UUIDS[gpu])))
     if len(pids)<6 and free>=4096:break
     write(work/'gpu_wait.json',dict(status='waiting',physical_gpu=gpu,compute_pids=pids,free_mib=free,need_free_mib=4096,updated_unix=time.time()))
     fcntl.flock(admission,fcntl.LOCK_UN);admission.close();admission=None;time.sleep(20)
   with (work/(mode+'.log')).open('x') as log:
    child=subprocess.Popen([PY,'-B','-u',__file__,mode,arm,str(gpu)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    write(work/'controller_status.json',dict(status=mode,controller_pid=os.getpid(),child_pid=child.pid,physical_gpu=gpu,updated_unix=time.time()))
    if admission is not None:
     deadline=time.time()+180
     while child.poll() is None:
      lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines()
      if child.pid in [int(x.split(',')[1]) for x in lines if x.startswith(UUIDS[gpu])]:break
      if time.time()>deadline:child.terminate();raise TimeoutError(f'{mode} phase did not become GPU resident')
      time.sleep(1)
     fcntl.flock(admission,fcntl.LOCK_UN);admission.close();admission=None
    rc=child.wait()
   write(work/(mode+'_exitcode.json'),dict(exitcode=rc,pid=child.pid));assert rc==0,mode+' failed; downstream blocked'
  assert read(work/'completion.json')['complete'];write(work/'controller_status.json',dict(status='complete',controller_pid=os.getpid(),physical_gpu=gpu))
if __name__=='__main__':
 try:
  if sys.argv[1]=='prepare':prepare()
  elif sys.argv[1]=='controller':controller(sys.argv[2],int(sys.argv[3]))
  else:phase(sys.argv[2],sys.argv[1],int(sys.argv[3]))
 except BaseException as e:
  target=RUN/sys.argv[2] if len(sys.argv)>2 else RUN
  if target.exists():write(target/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),pid=os.getpid(),time=time.time()))
  raise

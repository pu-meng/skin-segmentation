import os,sys,json,copy,csv,hashlib,shutil,subprocess,time,fcntl,types,io,traceback
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
OPS=ROOT/'operations/a1_skipc_v05_20261107'
RUN=Path('/media/8T/PuMengYu/experiments/skin_segmentation/A1_SkipC_v05_FRESH300_20261107')
REF=Path('/media/8T/PuMengYu/experiments/skin_segmentation/DecoderDualA_RESUME100to300_20261106/protocol.json')
UUIDS=['GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde','GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b']
sys.path[:0]=[str(OPS),str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
def write(p,o):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(o,ensure_ascii=False,indent=2));t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def imports():
 from pretrained_unet2d import prepare_imports
 prepare_imports('mambavision_t');prepare_imports('tinyvim_s')
 import torch
 import ham_fast_runner as runner
 from ham_fast_pipeline import make_fast_loaders
 from skip_c import build_candidate,SkipC
 return torch,runner,make_fast_loaders,build_candidate,SkipC
def disk():
 assert shutil.disk_usage(RUN).free>8*2**30,'result disk below8GiB; no cleanup'
 assert shutil.disk_usage(OPS).free>8*2**30,'home disk below8GiB'
def verify():
 p=json.loads((RUN/'protocol.json').read_text())
 for f,h in p['protected_sha256'].items():assert sha(f)==h,'source drift '+f
 assert sha(p['config']['data']['split_file'])==p['config']['data']['split_sha256'];disk();return p
def digest(sd):
 h=hashlib.sha256()
 for n,t in sorted(sd.items()):
  h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def prepare():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 assert not RUN.exists(),'refuse existing run'
 torch,runner,_,build,SkipC=imports();torch.set_num_threads(1)
 old=json.loads(REF.read_text());cfg=copy.deepcopy(old['config']);assert cfg['training']['epochs']==300 and cfg['training']['batch_size']==8
 for f,h in old['source_sha256'].items():assert sha(ROOT/f)==h,'old source drift '+f
 cfg['evaluation']['test_evaluation']='fixed validation best after300; test2020 none and mirror4; no test selection'
 api=runner.baseline_api();api.seed_all(43)
 from source.decoder_dual_a_20261106.candidate import build_candidate as base
 ref=base(True);refsd={k:v.clone() for k,v in ref.state_dict().items() if not k.startswith('decoder.e3.')};rng=torch.get_rng_state().clone();del ref
 api.seed_all(43);m=build(True);assert torch.equal(rng,torch.get_rng_state())
 kept={k:v for k,v in m.state_dict().items() if not k.startswith('decoder.e3.')}
 assert kept.keys()==refsd.keys() and all(torch.equal(v,refsd[k]) for k,v in kept.items())
 assert all(p.requires_grad for p in m.parameters());assert type(m.decoder.e3) is SkipC
 buf=io.BytesIO();torch.save(m.state_dict(),buf);buf.seek(0);m.load_state_dict(torch.load(buf,map_location='cpu',weights_only=False),strict=True)
 initial=digest(m.state_dict());params=sum(p.numel() for p in m.parameters());retained_count=len(kept);del m,refsd,kept,buf
 torch.manual_seed(51);e=SkipC();x=torch.randn(2,64,64,64,requires_grad=True);g=torch.randn_like(x,requires_grad=True);gc=g.detach().clone();z=e.components(x,g);y,d=e(x,g)
 torch.testing.assert_close(z['p'],torch.full_like(z['p'],1/3),rtol=0,atol=0)
 torch.testing.assert_close(z['F'],x,rtol=1e-6,atol=1e-6)
 torch.testing.assert_close(z['As'],torch.full_like(z['As'],.95),rtol=0,atol=1e-7)
 torch.testing.assert_close(z['gamma'],torch.tensor(.01),rtol=1e-6,atol=1e-8)
 torch.testing.assert_close(y,z['As']*z['F']+z['gamma']*z['Ad']*z['Delta'],rtol=0,atol=0)
 assert d is g and torch.equal(g,gc)
 zg=e.components(x,g+1)
 for k in ['p','F','As','Delta','gamma']:assert torch.equal(z[k],zg[k]),k
 diff=y-(zg['As']*zg['F']+zg['gamma']*zg['Ad']*zg['Delta'])
 torch.testing.assert_close(diff,z['gamma']*(z['Ad']-zg['Ad'])*z['Delta'],rtol=1e-3,atol=5e-7)
 only=z['As']*z['F']+z['gamma']*z['Ad'].detach()*z['Delta']
 assert torch.autograd.grad(only.sum(),g,allow_unused=True,retain_graph=True)[0] is None
 y.square().mean().backward();assert g.grad is not None and g.grad.abs().sum()>0 and torch.isfinite(g.grad).all()
 grads={n:float(p.grad.norm()) for n,p in e.named_parameters()};assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in e.parameters())
 e2=SkipC();e2.load_state_dict(e.state_dict(),strict=True);torch.testing.assert_close(e2(x,g)[0],y,rtol=0,atol=0)
 assert not torch.cuda.is_initialized();RUN.mkdir(parents=True);disk()
 sources={str(ROOT/f):h for f,h in old['source_sha256'].items()};sources[str(REF)]=sha(REF)
 # Capture all loaded project Python, plus exact new sources/test runtime.
 for mod in list(sys.modules.values()):
  f=getattr(mod,'__file__',None)
  if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py') and Path(f).exists():sources[str(Path(f).resolve())]=sha(f)
 for rel in ['operations/nnunet_mirror_validation/run_validation.py','operations/nnunet_mirror_validation/mirror_core.py','source/bottleneck_mamba/final_test_pipeline.py']:
  sources[str(ROOT/rel)]=sha(ROOT/rel)
 for f in OPS.glob('*'):
  if f.suffix in ['.py','.sh','.md']:sources[str(f)]=sha(f)
 protocol=dict(identity='A1 + full e3 replacement SkipC v0.5 fresh300',config=cfg,initialization='A1 original seed42 build_candidate(pretrained=True), stage fork45 retained; new e3 fork51; ImageNet only; no HAM checkpoint',initial_state_sha256=initial,parameters=params,protected_sha256=sources,parent_protocol=str(REF),contract='300 train epochs /873 steps, val1010 each epoch; earliest strict min val_loss; freeze best, CPU strict, test2020 none + mirror4 and artifact audits',implementation_choices={'gaussian':'fixed normalized separable5x5 sigma1 reflect2 stride1 depthwise; same kernel twice','local_padding':'DW3 zero1; PW1 biasTrue; no norm/activation','spatial_initialization':'PyTorch default nonzero Conv2d, no zero last layer','retain_bias':'log17 -> As .95','frequency_head':'zeros -> p1/3, F=X within FP32 tolerance','gamma':'softplus(log(expm1(.01)))','train_scope':'all parameters; no new loss or auxiliary boundary loss','loader':'workers2 fastval8 potentialbest/finalval1 inherited engineering','test':'batch1 FP32 noTTA and inverseflipped4view logits average [[],[2],[3],[2,3]]'},scheduling={'shared_gpu_authorized':False,'preference':[1,0],'wait_hours':72,'poll_seconds':60,'idle_confirmation_seconds':10,'same_gpu_reservations':{'0':695444,'1':695445},'condition':'no compute apps, used<500MiB, utilization<5%; same GPU SGCD controller reservation must have exited; other card never a dependency','phase_timeout_hours':168},retention='only inherited best/latest/final and fixed best snapshot; no per-epoch weight archives; no deletion of any old files',historical_comparison='existing A1 only; compare budgets explicitly; no new control arms')
 protocol['scheduling']['external_dispatch_file']=str(RUN/'dispatch.json')
 protocol['scheduling']['external_dispatch_policy']='mutable per-card allowlist; initial empty because A2/A3/A4 continuation reservations take priority; parent scheduler releases only unreserved GPU(s); no cross-card dependency'
 write(RUN/'protocol.json',protocol)
 write(RUN/'dispatch.json',dict(allowed_gpus=[],reason='A2/A3/A4 latest-to-total300 priority reservations pending external dispatcher',schema='set allowed_gpus to subset of [1,0] after same-card priority reservations are released; never edit protocol'))
 for f,h in sources.items():
  dst=RUN/'source_snapshot'/Path(f).relative_to('/')
  dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dst);assert sha(dst)==h
 write(RUN/'cpu_acceptance.json',dict(passed=True,device='server CPU; CUDA hidden',parameters=params,retained_tensors=retained_count,retained_A1_exact=True,cpu_rng_preserved=True,full_model_strict_roundtrip=True,module_shape=[2,64,64,64],frequency_equal_thirds=True,F_identity_max_error=float((z['F']-x).abs().max()),As=.95,gamma=float(z['gamma']),D_same_object_unchanged=True,formula_exact=True,G_only_via_Ad=True,G_first_backward_norm=float(g.grad.norm()),all_parameter_gradient_norms=grads,module_strict_reload=True,cuda_initialized=False,scope='module forward/backward + full candidate construction/state pairing/strict; NOT full CPU scan/backward or GPU preflight',protocol_sha256=sha(RUN/'protocol.json')))
 print('CPU_ACCEPTED',params,initial,flush=True)
def gpu_mapping(gpu):
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
 assert any(UUIDS[gpu] in l and f', {os.getpid()},' in l for l in apps.splitlines()),apps
 return apps
def phase(mode,gpu):
 p=verify();torch,runner,loaders_fn,build,_=imports();torch.set_num_threads(4);api=runner.baseline_api();api.seed_all(43)
 model=build(mode!='test');assert digest(model.state_dict())==p['initial_state_sha256'] if mode!='test' else True
 if mode=='test':return test(torch,runner,model,p,gpu)
 assert not torch.cuda.is_initialized();torch.cuda.set_device(0);model.cuda();apps=gpu_mapping(gpu);stage=RUN/('preflight' if mode=='preflight' else 'run');stage.mkdir()
 loaders,g=loaders_fn(p['config'],api,workers=2,validation_batch=8);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8);crit=api.BceDiceLoss().cuda()
 info=dict(model=p['identity'],pid=os.getpid(),seed=43,gpu_uuid=UUIDS[gpu],physical_gpu=gpu,protocol_sha256=sha(RUN/'protocol.json'),split_sha256=p['config']['data']['split_sha256'],source_sha256=p['protected_sha256'],fresh_start=True,training=p['config']['training'],engineering={'workers':2,'fast_val':8,'precise_val':1})
 write(stage/'provenance.json',info);write(stage/'weight_loading_audit.json',model.encoder.loading_audit)
 def hook(mod,ins,out):
  assert tuple(ins[0].shape[1:])==(64,64,64) and tuple(ins[1].shape[1:])==(64,64,64)
  assert out[1] is ins[1]
  if not (stage/'real_e3_boundary.json').exists():write(stage/'real_e3_boundary.json',dict(X=list(ins[0].shape),G=list(ins[1].shape),Y=list(out[0].shape),D_same_object=True))
 handle=model.decoder.e3.register_forward_hook(hook)
 try:
  if mode=='preflight':
   proof=runner.preflight(model,loaders,crit,opt,torch.device('cuda:0'),stage,[]);proof.update(pid=os.getpid(),gpu_uuid=UUIDS[gpu],physical_gpu=gpu,mapping=apps,protocol_sha256=sha(RUN/'protocol.json'));write(stage/'acceptance.json',proof);print('GPU_REAL_BATCH8_PREFLIGHT_PASSED',flush=True)
  else:
   assert json.loads((RUN/'preflight/acceptance.json').read_text())['status']=='passed'
   original=opt.step;updates=0
   def step(self,*a,**kw):
    nonlocal updates
    if updates%873==0:disk()
    first=updates==0
    if first:
     grads=[q.grad for q in model.parameters() if q.requires_grad];assert all(v is not None for v in grads);assert bool(torch.stack([torch.isfinite(v).all() for v in grads]).all());before=model.decoder.e3.theta.detach().clone()
    result=original(*a,**kw);updates+=1
    if first:
     torch.cuda.synchronize();after=model.decoder.e3.theta.detach().clone();assert not torch.equal(before,after)
     write(RUN/'first_optimizer_update.json',dict(pid=os.getpid(),gpu_uuid=UUIDS[gpu],mapping=gpu_mapping(gpu),batch=8,input=[8,3,256,256],epoch=1,step=1,finite_gradients=True,theta_before=before.item(),theta_after=after.item(),numerical_update=True,protocol_sha256=sha(RUN/'protocol.json')));print('FIRST_FORMAL_OPTIMIZER_UPDATE',os.getpid(),UUIDS[gpu],flush=True)
    return result
   opt.step=types.MethodType(step,opt)
   runner.train(model,loaders,crit,opt,torch.device('cuda:0'),stage,p['config'],info,g,resume=False)
   audit=json.loads((stage/'artifact_audit.json').read_text());assert audit['training_complete'] and audit['validation_artifacts_complete'] and audit['epochs']==300;verify()
 finally:
  handle.remove()
  for l in loaders.values():
   if hasattr(l,'close'):l.close()
def test(torch,runner,model,p,gpu):
 run=RUN/'run';rows=list(csv.DictReader((run/'history.csv').open()));assert [int(r['epoch']) for r in rows]==list(range(1,301))
 audit=json.loads((run/'artifact_audit.json').read_text());assert audit['training_complete'] and audit['validation_artifacts_complete']
 win=min(rows,key=lambda r:float(r['val_loss']));old=json.loads((run/'validation/summary.json').read_text());assert int(win['epoch'])==old['selected_epoch']
 src=run/'checkpoints/best.pth';assert old['checkpoint_sha256']==sha(src);out=RUN/'final_test';out.mkdir();snap=out/'fixed_best.pth';h=sha(src);shutil.copy2(src,snap);assert sha(src)==sha(snap)==h
 model.load_state_dict(torch.load(snap,map_location='cpu',weights_only=False),strict=True);assert not torch.cuda.is_initialized();write(out/'cpu_load_audit.json',dict(strict=True,checkpoint_sha256=h,selected_epoch=int(win['epoch']),cuda_initialized=False))
 from torch.utils.data import DataLoader
 import final_test_pipeline as pipeline
 from operations.nnunet_mirror_validation.run_validation import Predictor
 api=runner.baseline_api();api.seed_all(43);data=api.HAMDataset(Path(p['config']['data']['split_file']).parent,'test');ids={r['image_id'] for r in data.rows};assert len(data)==len(ids)==2020 and not data.training
 loader=DataLoader(data,batch_size=1,shuffle=False,num_workers=0);torch.cuda.set_device(0);model.cuda().eval();gpu_mapping(gpu);summaries={}
 for mode in ['none','mirror4']:
  disk();target=out/mode;assert not target.exists();pred=Predictor(model,mode=='mirror4').eval();print('START_TEST',mode,os.getpid(),UUIDS[gpu],flush=True)
  result=api.evaluate(pred,loader,api.BceDiceLoss().cuda(),torch.device('cuda:0'),target);assert len(pred.times)==2020 and pred.verify_remaining==0
  result.update(split='test',dataset='HAM10000',model=p['identity'],selected_epoch=int(win['epoch']),checkpoint=str(snap),checkpoint_sha256=h,split_sha256=p['config']['data']['split_sha256'],tta=mode=='mirror4',training_epochs=300,test_used_for_checkpoint_selection=False,physical_gpu=gpu,gpu_uuid=UUIDS[gpu])
  a=pipeline.validate_outputs(target,ids,result);assert a['complete'];write(target/'summary.json',result);write(target/'artifact_audit.json',a);pred.close();summaries[mode]=result
 assert sha(src)==sha(snap)==h;verify();write(RUN/'completion.json',dict(complete=True,epochs=300,validation_images=1010,test_images_per_mode=2020,selected_epoch=int(win['epoch']),checkpoint_sha256=h,results=summaries));print('TRAIN_VALIDATION_TEST_COMPLETE',flush=True)
def reservation_alive(pid):
 try:return 'HIC循环适配/controller.py' in Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\x00',' ')
 except FileNotFoundError:return False
def idle(gpu,p):
 dispatch=json.loads((RUN/'dispatch.json').read_text());allowed=dispatch['allowed_gpus'];assert isinstance(allowed,list) and all(type(g) is int and g in [0,1] for g in allowed)
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 stat=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 vals=next(l.split(',') for l in stat.splitlines() if l.startswith(UUIDS[gpu]));reserved=reservation_alive(p['scheduling']['same_gpu_reservations'][str(gpu)])
 return gpu in allowed and UUIDS[gpu] not in apps and int(vals[1])<500 and int(vals[2])<5 and not reserved,dict(gpu=gpu,uuid=UUIDS[gpu],external_dispatch_allowed=gpu in allowed,dispatch_reason=dispatch.get('reason'),memory_used_MiB=int(vals[1]),utilization=int(vals[2]),reservation_alive=reserved,apps=apps)
def controller():
 p=verify()
 with (RUN/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(RUN/'controller.pid').write_text(str(os.getpid())+'\n')
  deadline=time.time()+p['scheduling']['wait_hours']*3600;gpu=None
  while time.time()<deadline:
   verify();states=[]
   for candidate in p['scheduling']['preference']:
    free,s=idle(candidate,p);states.append(s)
    if free:
     time.sleep(p['scheduling']['idle_confirmation_seconds']);free,s=idle(candidate,p)
     if free:gpu=candidate;break
   if gpu is not None:break
   state=dict(status='waiting_idle_gpu_NOT_TRAINED',pid=os.getpid(),deadline_unix=deadline,updated_at=time.time(),resources=states);write(RUN/'controller_status.json',state);print(json.dumps(state,ensure_ascii=False),flush=True);time.sleep(p['scheduling']['poll_seconds'])
  if gpu is None:raise TimeoutError('72h idle wait expired; no training launched')
  # Single controller remains alive through both train and fixed-best test.
  for phase_name in ['preflight','train','test']:
   verify();env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=UUIDS[gpu],EXPERIMENT_PHYSICAL_GPU=str(gpu),EXPERIMENT_GPU_UUID=UUIDS[gpu])
   with (RUN/(phase_name+'.log')).open('x') as log:
    child=subprocess.Popen([sys.executable,'-B','-u',__file__,phase_name,str(gpu)],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    write(RUN/'controller_status.json',dict(status=phase_name,pid=os.getpid(),child_pid=child.pid,gpu=gpu,gpu_uuid=UUIDS[gpu],updated_at=time.time()));started=time.monotonic()
    for line in child.stdout:
     print(line,end='',flush=True);log.write(line);log.flush()
     if time.monotonic()-started>p['scheduling']['phase_timeout_hours']*3600:raise TimeoutError('phase runtime bound exceeded')
    rc=child.wait();write(RUN/(phase_name+'_exitcode.json'),dict(exitcode=rc,pid=child.pid));assert rc==0,phase_name+' failed; downstream blocked'
  assert json.loads((RUN/'completion.json').read_text())['complete'];write(RUN/'controller_status.json',dict(status='complete',pid=os.getpid(),gpu=gpu))
if __name__=='__main__':
 try:
  if sys.argv[1]=='prepare':prepare()
  elif sys.argv[1]=='controller':controller()
  else:phase(sys.argv[1],int(sys.argv[2]))
 except BaseException as e:
  if RUN.exists():write(RUN/'failure.json',dict(error=repr(e),pid=os.getpid(),traceback=traceback.format_exc(),time=time.time()))
  raise

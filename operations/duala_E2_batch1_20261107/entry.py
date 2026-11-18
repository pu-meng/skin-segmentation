import os,sys,json,copy,csv,hashlib,time,subprocess,fcntl,shutil,types
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation');OPS=ROOT/'operations/duala_E2_batch1_20261107';RUN=Path('/media/8T/PuMengYu/experiments/skin_segmentation/DualA_E2_FirstBatch100_20261107');REF=Path('/media/8T/PuMengYu/experiments/skin_segmentation/DecoderDualA_RESUME100to300_20261106')
sys.path[:0]=[str(OPS),str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
from source.e123_candidate_a_20261106.entry import runner,make_fast_loaders,torch
from candidates import build,E2,FourierCorrection,NAMES
from source.decoder_dual_a_20261106.candidate import build_candidate as base
ARMS=['NoE2','Input','Cross','AFNO'];BATCHES=[['NoE2','Input'],['Cross','AFNO']]
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2));tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(sd):
 h=hashlib.sha256()
 for n,t in sorted(sd.items()):h.update(n.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def make(arm,pretrained=True):runner.baseline_api().seed_all(42);return build(arm,pretrained)
def prepare():
 assert not RUN.exists(),'refuse existing run';torch.set_num_threads(1);p=json.loads((REF/'protocol.json').read_text());cfg=copy.deepcopy(p['config']);cfg['training'].update(epochs=100,total_optimizer_updates=87300,image_exposures=100*cfg['data']['counts']['train']);assert cfg['training']['updates_per_epoch']==873
 for f,h in p['source_sha256'].items():assert sha(ROOT/f)==h,'source drift '+f
 assert sha(cfg['data']['split_file'])==cfg['data']['split_sha256']
 runner.baseline_api().seed_all(42);reference=base(True);ref={k:v for k,v in reference.state_dict().items() if not k.startswith('decoder.e2.')};common=digest(ref);records={};models={}
 for arm in ARMS:
  m=make(arm);sd=m.state_dict();kept={k:v for k,v in sd.items() if not k.startswith('decoder.e2.')};assert digest(kept)==common;assert all(torch.equal(v,ref[k]) for k,v in kept.items());m.load_state_dict(copy.deepcopy(sd),strict=True);assert all(t.requires_grad for t in m.encoder.parameters());records[arm]={'parameters':sum(v.numel() for v in m.parameters()),'initial_state_sha256':digest(sd),'retained_tensors':len(kept),'retained_equal':True};models[arm]=m
 # CPU checks only use actual local/fusion/AFNO paths; CUDA Mamba is checked in the formal first batch.
 z=torch.randn(2,256,8,8,requires_grad=True);l=torch.randn(2,128,8,8,requires_grad=True);g=torch.randn_like(l,requires_grad=True)
 for arm in ['Input','Cross','AFNO']:
  e=models[arm].decoder.e2;e.fuse(z,l,g).square().mean().backward();assert all(torch.isfinite(t.grad).all() for t in [z,l,g] if t.grad is not None)
  for t in [z,l,g]:t.grad=None
 f=FourierCorrection();x=torch.randn(2,128,8,8,requires_grad=True);y=f(x);y.square().mean().backward();assert y.shape==x.shape and torch.isfinite(x.grad).all();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in f.parameters())
 for prefix in ['pre.','local.','output.','gamma']:
  a={k:v for k,v in models['Input'].decoder.e2.state_dict().items() if k.startswith(prefix)}
  for arm in ['Cross','AFNO']:
   b=models[arm].decoder.e2.state_dict();assert all(torch.equal(v,b[k]) for k,v in a.items())
 assert not torch.cuda.is_initialized();rows=list(csv.DictReader((REF/'DualA/train/history.csv').open()));assert len(rows)==300;prefix=rows[:100];best=min(prefix,key=lambda r:float(r['val_loss']));assert int(best['epoch'])==39
 RUN.mkdir()
 for arm in ARMS:(RUN/arm).mkdir()
 with (RUN/'reference100_history.csv').open('w',newline='') as out:
  w=csv.DictWriter(out,fieldnames=list(prefix[0]));w.writeheader();w.writerows(prefix)
 protected={str(ROOT/f):h for f,h in p['source_sha256'].items()};protected.update({str(REF/'protocol.json'):sha(REF/'protocol.json'),str(REF/'DualA/train/history.csv'):sha(REF/'DualA/train/history.csv'),str(REF/'DualA/train/checkpoints/best.pth'):sha(REF/'DualA/train/checkpoints/best.pth')})
 files={str(f):sha(f) for f in OPS.glob('*.py')};files[str(OPS/'run.sh')]=sha(OPS/'run.sh')
 write(RUN/'protocol.json',{'identity':'DualA fixed decoder/skip; replace E2 only; first batch validation100','config':cfg,'arms':NAMES,'batches':BATCHES,'records':records,'common_initial_state_sha256':common,'reference_reuse':{'path':str(REF/'DualA/train/history.csv'),'prefix_rows':100,'selected_epoch':39,'source_unchanged':True,'initialization':'same build_base seed42 and DualStage fork45, ImageNet only; no HAM checkpoint used'},'protected_sha256':protected,'implementation_sha256':files,'official_afno':json.loads((OPS/'afno_source.json').read_text()),'gpu_uuid':'GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde','physical_gpu':0,'precision':'FP32','seed':42,'new_E2_seed':48,'cross_projection_seed':49,'afno_seed':50,'shared_gpu_authorized':True,'contract':'100 epochs + validation selected best/full val artifacts; no test, no automatic continuation; original scheduler T_max50; no encoder freezing','implementation':'shared LN512/PW512-256; split128/128; local LN/DW7/GELU/PW; global normalized Mamba1 correction or official AFNO correction; input channel softmax or one-way 4head localQ/globalKV; shared LN/GELU/PW256-512; gamma.01, single X residual; no FFN','test_data_accessed':False})
 write(RUN/'cpu_acceptance.json',{'passed':True,'arms':records,'strict_reload':True,'local_and_output_initialization_equal':True,'afno_cpu_backward':True,'fusion_cpu_backward':True,'cuda_initialized':False,'scope':'construction, shared retained state and fusion/AFNO paths; full batch8 CUDA gradients at launch'});print('CPU_ACCEPTED',records,flush=True)
def verify():
 p=json.loads((RUN/'protocol.json').read_text())
 for f,h in {**p['protected_sha256'],**p['implementation_sha256']}.items():assert sha(f)==h,'source drift '+f
 assert sha(p['config']['data']['split_file'])==p['config']['data']['split_sha256'];assert shutil.disk_usage(RUN).free>8*2**30
 return p
def wait_for(arms,file,seconds=600):
 end=time.monotonic()+seconds
 while not all((RUN/a/file).exists() for a in arms):
  assert not any((RUN/a/'failure.json').exists() for a in arms),'peer failed'
  if time.monotonic()>end:raise TimeoutError(file)
  time.sleep(1)
def train(arm):
 p=verify();torch.set_num_threads(4);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.20);m=make(arm);assert digest(m.state_dict())==p['records'][arm]['initial_state_sha256'];m.cuda();api=runner.baseline_api();loaders,g=make_fast_loaders(p['config'],api,workers=2,validation_batch=8);criterion=api.BceDiceLoss().cuda();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8);stage=RUN/arm/'run';stage.mkdir();batch=next(b for b in BATCHES if arm in b)
 info={'model':NAMES[arm],'arm':arm,'seed':42,'pid':os.getpid(),'gpu_uuid':p['gpu_uuid'],'physical_gpu':0,'parameters':p['records'][arm]['parameters'],'protocol_sha256':sha(RUN/'protocol.json'),'split_sha256':p['config']['data']['split_sha256'],'source_sha256':{**p['protected_sha256'],**p['implementation_sha256']},'fresh_start':True,'test_data_accessed':False,'training':p['config']['training']};write(stage/'provenance.json',info);write(stage/'weight_loading_audit.json',m.encoder.loading_audit);write(RUN/arm/'ready.json',{'pid':os.getpid()});wait_for(batch,'ready.json')
 original=opt.step;seen=False
 def step(self,*a,**kw):
  nonlocal seen
  if seen:return original(*a,**kw)
  grads=[t.grad for t in m.parameters() if t.requires_grad];missing=[n for n,t in m.named_parameters() if t.requires_grad and t.grad is None];assert not missing,missing;assert bool(torch.stack([torch.isfinite(t).all() for t in grads]).all())
  param=m.decoder.e3.s.gamma if arm=='NoE2' else m.decoder.e2.gamma;before=param.detach().clone();res=original(*a,**kw);torch.cuda.synchronize();assert not torch.equal(before,param)
  mapping=subprocess.getoutput('nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader');assert any(p['gpu_uuid'] in l and f', {os.getpid()},' in l for l in mapping.splitlines())
  write(RUN/arm/'first_optimizer_update.json',{'pid':os.getpid(),'gpu_uuid':p['gpu_uuid'],'batch':8,'epoch':1,'step':1,'finite_gradients':True,'missing_gradients':missing,'numerical_update':True,'max_delta':float((before-param).abs().max()),'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20});seen=True;print('FORMAL_BATCH8_UPDATE',arm,os.getpid(),flush=True);wait_for(batch,'first_optimizer_update.json');return res
 opt.step=types.MethodType(step,opt)
 try:
  runner.train(m,loaders,criterion,opt,torch.device('cuda:0'),stage,p['config'],info,g,resume=False);verify();audit=json.loads((stage/'artifact_audit.json').read_text());assert audit['training_complete'] and audit['validation_artifacts_complete'] and audit['epochs']==100
  write(RUN/arm/'completion.json',{'complete':True,'epochs':100,'validation_complete':True,'test_data_accessed':False})
 finally:
  for l in loaders.values():
   if hasattr(l,'close'):l.close()
def slot(slot):
 arms=['NoE2','AFNO'] if slot=='A' else ['Input','Cross']
 with (RUN/('slot'+slot+'.lock')).open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);state={'status':'running','pid':os.getpid(),'arms':arms,'completed':[]};write(RUN/('slot'+slot+'.json'),state)
  for i,arm in enumerate(arms):
   if i:state.update(status='waiting_for_batch1',current=arm);write(RUN/('slot'+slot+'.json'),state);wait_for(BATCHES[0],'completion.json',24*3600)
   env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
   with (RUN/arm/'train.log').open('x') as log:
    child=subprocess.Popen([sys.executable,'-B','-u',__file__,'train',arm],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1);state.update(status='training',current=arm,child_pid=child.pid);write(RUN/('slot'+slot+'.json'),state)
    for line in child.stdout:print(line,end='',flush=True);log.write(line);log.flush()
    code=child.wait()
   state['last_exitcode']=code
   if code:state.update(status='failed');write(RUN/('slot'+slot+'.json'),state);return code
   assert json.loads((RUN/arm/'completion.json').read_text())['complete'];state['completed'].append(arm);write(RUN/('slot'+slot+'.json'),state)
  state.update(status='complete',current=None);write(RUN/('slot'+slot+'.json'),state);return 0
if __name__=='__main__':
 if sys.argv[1]=='slot':sys.exit(slot(sys.argv[2]))
 try:
  if sys.argv[1]=='prepare':prepare()
  else:train(sys.argv[2])
 except BaseException as e:
  if len(sys.argv)>2:write(RUN/sys.argv[2]/'failure.json',{'pid':os.getpid(),'error':repr(e)})
  raise

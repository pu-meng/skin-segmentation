import os,sys,json,copy,hashlib,time,subprocess,fcntl,shutil
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation');OPS=ROOT/'operations/candidateA_E2_pair100_20261107';RUN=Path('/media/8T/PuMengYu/experiments/skin_segmentation/CandidateA_E2_Paired100_20261107')
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
from source.e123_candidate_a_20261106.entry import runner,build_candidate,make_fast_loaders,torch

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2));t.replace(p)
def digest(state):
 h=hashlib.sha256()
 for n,v in sorted(state.items()):h.update(n.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def make(arm,pretrained=True):
 runner.baseline_api().seed_all(42);model=build_candidate(pretrained)
 if arm=='NoE2':model.decoder.e2=torch.nn.Identity()
 return model
def prepare():
 assert not RUN.exists(),'refuse existing paired run';torch.set_num_threads(1)
 old=ROOT/'log/TinyViM_E123_CandidateA_E2Spectral_B3x3_HAM100_20261106/protocol.json';p=json.loads(old.read_text());cfg={k:copy.deepcopy(p[k]) for k in ['data','training','preprocessing','evaluation']};cfg['training'].update(epochs=100,total_optimizer_updates=100*cfg['training']['updates_per_epoch'],image_exposures=100*cfg['data']['counts']['train'])
 assert cfg['training']['updates_per_epoch']==873;assert sha(cfg['data']['split_file'])==cfg['data']['split_sha256']
 protected={str(ROOT/f):sha(ROOT/f) for f in p['source_sha256']};protected[str(old)]=sha(old)
 a=make('Full');b=make('NoE2');sa=a.state_dict();sb=b.state_dict();assert all(torch.equal(v,sa[n]) for n,v in sb.items());assert set(sa)-set(sb) and all(n.startswith('decoder.e2.') for n in set(sa)-set(sb));b.load_state_dict(copy.deepcopy(sb),strict=True);a.load_state_dict(copy.deepcopy(sa),strict=True)
 assert all(x.requires_grad for x in a.encoder.parameters()) and all(x.requires_grad for x in b.encoder.parameters());assert not torch.cuda.is_initialized()
 common=digest(sb);init={'Full':digest(sa),'NoE2':common};params={k:sum(p.numel() for p in m.parameters()) for k,m in [('Full',a),('NoE2',b)]}
 RUN.mkdir()
 for arm in ['Full','NoE2']:(RUN/arm).mkdir()
 write(RUN/'protocol.json',{'identity':'CandidateA Full versus E2 identity paired100','config':cfg,'seed':42,'arms':['Full','NoE2'],'parameters':params,'initial_state_sha256':init,'common_initial_state_sha256':common,'protected_sha256':protected,'source_sha256':p['source_sha256'],'entry_sha256':sha(__file__),'gpu_uuid':'GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde','physical_gpu':0,'precision':'FP32','encoder':'pretrained ImageNet, all trainable, not frozen','workers':2,'train_batch':8,'validation_batch':8,'potential_best_recheck_batch':1,'scheduler':'original CosineAnnealingLR T_max50 eta_min1e-5 unchanged; stop at100, not compressed schedule','completion_contract':'100 full epochs each; same validation minloss earliest best and full validation audit; no test; no automatic continuation; keep latest for authorized continuation','authorization':'screenshot paired original CandidateA versus only E2 identity, same initialization, first100 validation','shared_gpu':'existing processes protected; no pause or kill','test_data_accessed':False})
 write(RUN/'cpu_acceptance.json',{'passed':True,'retained_state_tensors_equal':len(sb),'only_removed_prefix':'decoder.e2.','strict_state_roundtrip':True,'encoder_trainable':True,'cuda_initialized':False,'parameters':params});print('PAIRED_CPU_ACCEPTED',params,common,flush=True)
def verify():
 m=json.loads((RUN/'protocol.json').read_text());assert sha(__file__)==m['entry_sha256']
 for f,h in m['protected_sha256'].items():assert sha(f)==h,'source drift '+f
 assert sha(m['config']['data']['split_file'])==m['config']['data']['split_sha256'];assert shutil.disk_usage(RUN).free>8*2**30
 return m
def barrier(name):
 end=time.monotonic()+600
 while not all((RUN/a/name).exists() for a in ['Full','NoE2']):
  assert not any((RUN/a/'failure.json').exists() for a in ['Full','NoE2']),'peer failed'
  if time.monotonic()>end:raise TimeoutError(name)
  time.sleep(.5)
def train(arm):
 m=verify();torch.set_num_threads(4);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.22);api=runner.baseline_api();model=make(arm);assert digest(model.state_dict())==m['initial_state_sha256'][arm];model.cuda();stage=RUN/arm/'run';stage.mkdir(parents=True);assert not any(stage.iterdir())
 loaders,g=make_fast_loaders(m['config'],api,workers=2,validation_batch=8);criterion=api.BceDiceLoss().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8)
 info={'model':'候选A完整原版' if arm=='Full' else '候选A仅E2恒等','arm':arm,'seed':42,'pid':os.getpid(),'physical_gpu':0,'gpu_uuid':m['gpu_uuid'],'protocol_sha256':sha(RUN/'protocol.json'),'split_sha256':m['config']['data']['split_sha256'],'source_sha256':m['source_sha256'],'parameters':m['parameters'][arm],'training':m['config']['training'],'fresh_start':True,'test_data_accessed':False}
 write(stage/'provenance.json',info);write(stage/'weight_loading_audit.json',model.encoder.loading_audit);write(RUN/arm/'ready.json',{'pid':os.getpid()});barrier('ready.json')
 original=opt.step;seen=False
 def step(*args,**kw):
  nonlocal seen
  if seen:return original(*args,**kw)
  grads=[p.grad for p in model.parameters() if p.requires_grad];assert all(x is not None for x in grads);assert bool(torch.stack([torch.isfinite(x).all() for x in grads]).all())
  p=model.decoder.e3.s.gamma;before=p.detach().clone();result=original(*args,**kw);torch.cuda.synchronize();assert not torch.equal(before,p)
  mapping=subprocess.getoutput('nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader');assert any(m['gpu_uuid'] in l and f', {os.getpid()},' in l for l in mapping.splitlines())
  write(RUN/arm/'first_optimizer_update.json',{'pid':os.getpid(),'gpu_uuid':m['gpu_uuid'],'epoch':1,'step':1,'batch':8,'finite_gradients':True,'numerical_update':True,'shared_parameter':'decoder.e3.s.gamma','max_delta':float((p-before).abs().max()),'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20});print('PAIRED_FORMAL_FIRST_UPDATE',arm,os.getpid(),flush=True);seen=True;barrier('first_optimizer_update.json');return result
 opt.step=step
 try:
  runner.train(model,loaders,criterion,opt,torch.device('cuda:0'),stage,m['config'],info,g,resume=False);a=json.loads((stage/'artifact_audit.json').read_text());assert a['training_complete'] and a['validation_artifacts_complete'] and a['epochs']==100;verify();write(RUN/arm/'completion.json',{'complete':True,'epochs':100,'validation_complete':True,'test_data_accessed':False})
 finally:
  for l in loaders.values():
   if hasattr(l,'close'):l.close()
def controller(arm):
 with (RUN/arm/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);state={'status':'starting','arm':arm,'controller_pid':os.getpid()};write(RUN/arm/'controller_status.json',state)
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
  with (RUN/arm/'train.log').open('x') as log:
   p=subprocess.Popen([sys.executable,'-B','-u',__file__,'train',arm],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1);state.update(status='training',child_pid=p.pid);write(RUN/arm/'controller_status.json',state)
   for line in p.stdout:print(line,end='',flush=True);log.write(line);log.flush()
   code=p.wait()
  state.update(status='complete' if code==0 else 'failed',exitcode=code);write(RUN/arm/'controller_status.json',state);return code
if __name__=='__main__':
 try:
  if sys.argv[1]=='prepare':prepare()
  elif sys.argv[1]=='train':train(sys.argv[2])
  else:sys.exit(controller(sys.argv[2]))
 except BaseException as e:
  if len(sys.argv)>2:write(RUN/sys.argv[2]/'failure.json',{'error':repr(e),'pid':os.getpid()})
  raise

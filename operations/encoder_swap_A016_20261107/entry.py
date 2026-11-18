import os,sys,json,copy,io,time,types,importlib.util,traceback,shutil,subprocess,fcntl,threading
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
OPS=Path(__file__).resolve().parent
RUN=Path('/media/8T/PuMengYu/experiments/skin_segmentation')/('EncoderSwap_A0_A1_A6_FRESH300_'+OPS.name.rsplit('_',1)[1])
PY='/home/PuMengYu/anaconda3/envs/medseg/bin/python'
sys.path[:0]=[str(OPS),str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
from encoder_swap_candidate import build_candidate,original,structure,JOBS,ENCODERS
TEMPLATE=ROOT/'operations/a1_skipc_v05_20261107/entry.py'
spec=importlib.util.spec_from_file_location('encoder_swap_frozen_runner',TEMPLATE);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
write=b.write;sha=b.sha;UUIDS=b.UUIDS
REF=Path('/media/8T/PuMengYu/experiments/skin_segmentation/A1_SkipC_v05_FRESH300_20261107/protocol.json')
def disk():
 for path in [ROOT,RUN]:assert shutil.disk_usage(path).free>8*2**30,'disk below8GiB '+str(path)
def verify(job=None):
 p=json.loads((RUN/'protocol.json').read_text())
 for f,h in p['protected_sha256'].items():assert sha(f)==h,'source drift '+f
 assert sha(p['config']['data']['split_file'])==p['config']['data']['split_sha256'];disk()
 if job is None:return p
 q=json.loads((RUN/job/'protocol.json').read_text());assert q['parent_sha256']==sha(RUN/'protocol.json');return q
def prepare():
 import torch
 import ham_fast_runner as runner
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not RUN.exists()
 torch.set_num_threads(1);api=runner.baseline_api();cfg=copy.deepcopy(json.loads(REF.read_text())['config'])
 assert cfg['training']['epochs']==300 and cfg['training']['T_max_epochs']==50 and cfg['training']['batch_size']==8
 RUN.mkdir();disk();records={};sizes=[];shapes={}
 for job in JOBS:
  api.seed_all(42);m=build_candidate(job,True);arm=job.split('__')[1];classes=structure(m,arm)
  initial=b.digest(m.state_dict());api.seed_all(42);ref=original(arm,False)
  kept={k:v for k,v in m.state_dict().items() if not k.startswith(('encoder.','align.'))}
  reference={k:v for k,v in ref.state_dict().items() if not k.startswith(('encoder.','align.'))}
  assert kept.keys()==reference.keys() and all(torch.equal(v,reference[k]) for k,v in kept.items())
  assert all(p.requires_grad for p in m.parameters());del ref,reference,kept
  stream=io.BytesIO();torch.save(m.state_dict(),stream);stream.seek(0);m.load_state_dict(torch.load(stream,map_location='cpu',weights_only=False),strict=True);del stream
  encoder=job.split('__')[0]
  if encoder not in shapes:
   m.eval()
   with torch.no_grad():
    x=torch.zeros(1,3,256,256);raw=m.raw_features(x);aligned=[a(f) for a,f in zip(m.align,raw)]
   shapes[encoder]={'raw':[list(t.shape) for t in raw],'aligned':[list(t.shape) for t in aligned]}
   assert [tuple(t.shape[1:]) for t in aligned]==[(64,64,64),(128,32,32),(256,16,16),(512,8,8)]
   del x,raw,aligned
  # Only new channel-align operators require CPU gradient diagnostics; do not
  # run the unchanged CUDA scan decoder/backward on a heavy CPU fallback.
  grad={}
  for i,layer in enumerate(m.align):
   t=torch.randn(2,layer.in_channels,4,4,requires_grad=True);layer(t).square().mean().backward()
   for n,p in layer.named_parameters():assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0;grad[str(i)+'.'+n]=p.grad.norm().item()
  size=sum(t.numel()*t.element_size() for t in m.state_dict().values());sizes.append(size)
  (RUN/job).mkdir();write(RUN/job/'weight_loading_audit.json',m.encoder.loading_audit)
  records[job]=dict(identity=job+' fresh300 seed42',initial_state_sha256=initial,state_bytes=size,parameters=sum(p.numel() for p in m.parameters()),classes=classes,retained_original_decoder_and_head_exact=True,full_state_strict_roundtrip=True,all_parameters_trainable=True,align_gradient_norms=grad,encoder_features=shapes[encoder])
  write(RUN/job/'cpu_acceptance.json',dict(passed=True,device='CPU CUDA hidden',cuda_initialized=torch.cuda.is_initialized(),**records[job]));assert not torch.cuda.is_initialized()
  print('CPU_ACCEPTED',job,size,flush=True);del m
 # Full retained latest(model+Adam states), best/final/fixed-best, both active
 # atomic latest temporary files, uncompressed prediction ceiling + figures.
 required=sum(sizes)*6+max(sizes)*6+len(JOBS)*2*2020*256*256+2*2**30
 available=shutil.disk_usage(RUN).free
 storage=dict(required_peak_bytes=required,available_bytes=available,reserve_bytes=8*2**30,model_state_bytes=sizes,formula='6*sum(state_bytes) retained +6*max(state_bytes) two latest temporaries +12*2*2020*256*256 prediction bytes +2GiB figures; plus8GiB guard',fits=available>=required+8*2**30)
 write(RUN/'storage_budget.json',storage);assert storage['fits'],'insufficient total9arm space; no cleanup'
 protected={str(TEMPLATE):sha(TEMPLATE),str(REF):sha(REF)}
 for mod in list(sys.modules.values()):
  f=getattr(mod,'__file__',None)
  if f and str(f).startswith(str(ROOT)) and f.endswith('.py') and Path(f).is_file():protected[str(Path(f).resolve())]=sha(f)
 for f in OPS.glob('*'):
  if f.suffix in ['.py','.sh']:protected[str(f)]=sha(f)
 for rel in ['operations/nnunet_mirror_validation/run_validation.py','operations/nnunet_mirror_validation/mirror_core.py','source/bottleneck_mamba/final_test_pipeline.py','source/comparison/ham_fast_pipeline.py']:
  protected[str(ROOT/rel)]=sha(ROOT/rel)
 protocol=dict(identity='RMT-T/RMT-S/SwiftFormer-S/MobileOne-S1 unfused x original A0/A1/A6 fresh300',config=cfg,jobs=records,protected_sha256=protected,initialization='official local classifier strict; MobileOne unfused training branches retained; full original A source decoder/head seed42 preserved; only encoder +1x1 align seed43 replaced; no HAM checkpoint loaded',contract='300epochs873steps261900updates seed42 batch8 FP32 AdamW BceDice; existing cosine T_max50; val1010 earliest strict min loss; fixed best CPUstrict test2020 none+mirror4; no test selection',engineering=dict(workers=2,fast_validation_batch=8,potential_best_and_final_validation_batch=1),scheduling=dict(shared_gpu_authorized=True,max_total_experiments_per_gpu=2,existing_tasks_count_as_slots=True,deadline_hours=336,poll_seconds=30,single_instance=True,per_gpu_slot_locks=True,missing3_priority=True,deadline_policy='No new phase or arm after deadline; never kill an already running phase',policy='GPU1 one new worker shares with existing SkipC now; GPU0 waits full incompatible A234 queue. Slot2 reserved for old queues including missing3; at arm boundaries yield chosen missing3 GPU until its entire queue audited, other GPU independent. When old reservations released allow two new workers. Each slot preflight->fresh300->both fixedbest tests+audits->next arm; no pair completion barrier'),storage=storage)
 write(RUN/'protocol.json',protocol)
 for job,r in records.items():write(RUN/job/'protocol.json',dict(**r,config=cfg,protected_sha256=protected,parent_sha256=sha(RUN/'protocol.json')))
 for f,h in protected.items():
  dst=RUN/'source_snapshot'/Path(f).relative_to('/');dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dst);assert sha(dst)==h
 write(RUN/'prepare_complete.json',dict(passed=True,arms=JOBS,count=len(JOBS),protocol_sha256=sha(RUN/'protocol.json'),cuda_initialized=False));print('ALL12_CPU_ACCEPTED',flush=True)
def bind(job):
 b.RUN=RUN/job;b.OPS=OPS;b.verify=lambda:verify(job);b.disk=disk

def phase(job,mode,gpu):
 import torch
 import ham_fast_runner as runner
 from ham_fast_pipeline import make_fast_loaders
 p=verify(job);bind(job);torch.set_num_threads(4);api=runner.baseline_api();api.seed_all(42)
 m=build_candidate(job,mode!='test');structure(m,job.split('__')[1]);assert all(t.requires_grad for t in m.parameters())
 if mode=='test':return b.test(torch,runner,m,p,gpu)
 assert b.digest(m.state_dict())==p['initial_state_sha256'];assert not torch.cuda.is_initialized()
 torch.cuda.set_device(0);m.cuda();mapping=b.gpu_mapping(gpu);stage=RUN/job/('preflight' if mode=='preflight' else 'run');stage.mkdir()
 loaders,g=make_fast_loaders(p['config'],api,workers=2,validation_batch=8);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8);crit=api.BceDiceLoss().cuda()
 info=dict(model=p['identity'],pid=os.getpid(),seed=42,gpu_uuid=UUIDS[gpu],physical_gpu=gpu,protocol_sha256=sha(RUN/job/'protocol.json'),split_sha256=p['config']['data']['split_sha256'],source_sha256=p['protected_sha256'],fresh_start=True,training=p['config']['training'],engineering={'workers':2,'fast_val':8,'precise_val':1})
 write(stage/'provenance.json',info);write(stage/'weight_loading_audit.json',m.encoder.loading_audit)
 def boundary(mod,args):
  fs=args[0];assert [tuple(t.shape[1:]) for t in fs]==[(64,64,64),(128,32,32),(256,16,16),(512,8,8)]
  if not (stage/'real_decoder_boundary.json').exists():write(stage/'real_decoder_boundary.json',dict(shapes=[list(t.shape) for t in fs],classes=structure(m,job.split('__')[1])))
 handle=m.decoder.register_forward_pre_hook(boundary)
 try:
  if mode=='preflight':
   proof=runner.preflight(m,loaders,crit,opt,torch.device('cuda:0'),stage,[]);proof.update(pid=os.getpid(),gpu_uuid=UUIDS[gpu],mapping=mapping,protocol_sha256=info['protocol_sha256']);write(stage/'acceptance.json',proof);print('REAL_BATCH8_PREFLIGHT_PASSED',job,flush=True)
  else:
   assert json.loads((RUN/job/'preflight/acceptance.json').read_text())['status']=='passed'
   original_step=opt.step;updates=0
   def step(self,*a,**kw):
    nonlocal updates
    if updates%873==0:disk()
    first=updates==0
    if first:
     pairs=[(n,q) for n,q in m.named_parameters() if q.requires_grad];assert all(q.grad is not None for n,q in pairs);assert bool(torch.stack([torch.isfinite(q.grad).all() for n,q in pairs]).all())
     probes={}
     for prefix in ['encoder.','align.','decoder.']:
      name,q=next((n,q) for n,q in reversed(pairs) if n.startswith(prefix) and bool(q.grad.abs().max()>0));probes[name]=q.detach().clone()
    result=original_step(*a,**kw);updates+=1
    if first:
     torch.cuda.synchronize();params=dict(m.named_parameters());delta={n:float((params[n]-v).abs().max()) for n,v in probes.items()};assert all(v>0 for v in delta.values())
     write(RUN/job/'first_optimizer_update.json',dict(pid=os.getpid(),gpu_uuid=UUIDS[gpu],mapping=b.gpu_mapping(gpu),batch=8,input=[8,3,256,256],epoch=1,step=1,finite_gradients=True,all_trainable_parameters_in_optimizer=True,nonzero_gradient_probes_max_delta=delta,numerical_update=True,protocol_sha256=info['protocol_sha256']));print('FIRST_FORMAL_OPTIMIZER_UPDATE',job,flush=True)
    return result
   opt.step=types.MethodType(step,opt);runner.train(m,loaders,crit,opt,torch.device('cuda:0'),stage,p['config'],info,g,resume=False)
   a=json.loads((stage/'artifact_audit.json').read_text());assert a['training_complete'] and a['validation_artifacts_complete'] and a['epochs']==300;verify(job)
 finally:
  handle.remove()
  for loader in loaders.values():
   if hasattr(loader,'close'):loader.close()
def read(p):return json.loads(Path(p).read_text()) if Path(p).exists() else {}
def upstream(gpu):
 missing=ROOT/'operations/missing3_resume300_20261107';claim=read(missing/'priority_accepted.json')
 if claim.get('gpu') not in [0,1]:return False,'missing3 has first claim; waiting its priority_accepted gpu'
 spec=importlib.util.spec_from_file_location('encoder_swap_existing_priority',missing/'entry.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 ready,reason=old.upstream_ready(gpu)
 if not ready:return False,reason
 if claim['gpu']!=gpu:return True,'own full upstream released; missing3 reserved other GPU'
 state=read(missing/'queue_status.json')
 if state.get('status')!='complete' or set(state.get('completed',[]))!={'AFull','ANoE2','DNoE2'}:return False,'same-card entire missing3 train+both test queue pending'
 for job in ['AFull','ANoE2','DNoE2']:
  c=read(missing/job/'completion.json');assert c.get('complete') and c['training_epochs']==300 and c['test_images_per_mode']==2020
  for phase_name in ['preflight','train','test']:assert read(missing/job/(phase_name+'_exitcode.json')).get('exitcode')==0
  for mode in ['none','mirror4']:assert read(Path(c['result_root'])/mode/'artifact_audit.json').get('complete') is True
 return True,'entire missing3 completed including all six test audits'
def resources(gpu):
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
 stats=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 vals=next(s.split(',') for s in stats.splitlines() if s.startswith(UUIDS[gpu]))
 pids=[int(l.split(',')[1]) for l in apps.splitlines() if l.startswith(UUIDS[gpu])]
 return dict(gpu=gpu,gpu_uuid=UUIDS[gpu],compute_pids=pids,used_mib=int(vals[1]),total_mib=int(vals[2]),free_mib=int(vals[2])-int(vals[1]),utilization=int(vals[3]),raw_compute=apps)
def admission(gpu,slot,secondary_ready,observed,primary_active):
 # A reserved old worker counts as one slot even during a phase gap.
 if slot==1 and not secondary_ready:return False,'second slot reserved for old same-card queue'
 if slot==1 and not primary_active:return False,'await first new worker resident formal training proof'
 if len(observed['compute_pids'])>=2:return False,'two experiment compute slots already occupied'
 if observed['free_mib']<6144:return False,'less than6GiB free; batch8 unchanged; wait or block at preflight'
 return True,'shared authorized; total maximum2 per GPU; utilization not a gate'

def compatible(gpu,slot):
 base=RUN.parent;old=ROOT/'operations'
 a234=read(old/'a234_resume300_20261107/queue_status.json')
 skip=read(base/'A1_SkipC_v05_FRESH300_20261107/controller_status.json')
 missing=read(old/'missing3_resume300_20261107/priority_accepted.json')
 # A234 and missing3 reject unrelated compute at every transition; leave them
 # untouched. SkipC checks idleness only on initial admission and can share.
 if gpu==0 and a234.get('status')!='complete':return False,'GPU0 A234 has incompatible no-sharing phase guard; wait entire old queue'
 if gpu==1 and slot==0 and skip.get('status')!='complete':return True,'share one added slot with current SkipC; its phase transitions permit sharing'
 # After own original queue releases, yield at the arm boundary until missing3
 # has acquired a card. Its chosen card stays exclusive until whole queue done.
 ready,why=upstream(gpu)
 return ready,why

def controller():
 verify();assert read(RUN/'prepare_complete.json').get('count')==12
 with (RUN/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);deadline=time.time()+336*3600
  state=dict(status='shared_authorized',pid=os.getpid(),started_unix=time.time(),deadline_unix=deadline,completed=[],assigned={},slots={});write(RUN/'controller_status.json',state)
  mutex=threading.Lock();failed=threading.Event();remaining=list(JOBS);active={};admit_locks=[threading.Lock(),threading.Lock()]
  def card(gpu,slot):
   key=f'{gpu}:{slot}'
   try:
    with (OPS/f'gpu{gpu}_slot{slot}.lock').open('a') as gpu_lock:
     fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
     while time.time()<deadline and not failed.is_set():
      with mutex:
       if not remaining:break
       first=active.get((gpu,0));primary_active=bool(first and (RUN/first/'first_optimizer_update.json').exists())
      ready,why=compatible(gpu,slot);observed=resources(gpu);allowed,why2=admission(gpu,slot,ready,observed,primary_active)
      if not ready or not allowed:
       with mutex:state['slots'][key]=dict(status='waiting',reason=why if not ready else why2,observed=observed,updated_unix=time.time());write(RUN/'controller_status.json',state)
       time.sleep(30);continue
      with mutex:
       if failed.is_set() or not remaining:break
       job=remaining.pop(0);active[(gpu,slot)]=job;state['assigned'][job]={'gpu':gpu,'slot':slot};write(RUN/'controller_status.json',state)
      for mode in ['preflight','train','test']:
       assert time.time()<deadline,'queue deadline expired; no new phase launched';verify(job)
       env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=UUIDS[gpu],EXPERIMENT_PHYSICAL_GPU=str(gpu),EXPERIMENT_GPU_UUID=UUIDS[gpu],OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
       with (RUN/job/(mode+'.log')).open('x') as log:
        with admit_locks[gpu]:
         observed=resources(gpu);assert len(observed['compute_pids'])<2,'two compute slots occupied; no third admitted'
         if mode=='preflight':assert observed['free_mib']>=6144,'preflight resource block; batch8 unchanged'
         write(RUN/job/(mode+'_shared_resources.json'),dict(authorization='explicit shared2slots perGPU incl old tasks',slot=slot,observed=observed))
         child=subprocess.Popen([PY,'-B','-u',__file__,mode,job,str(gpu)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
         with mutex:state['slots'][key]=dict(status=mode,job=job,child_pid=child.pid,gpu_uuid=UUIDS[gpu]);write(RUN/'controller_status.json',state)
         # Hold local admission lock until this child is actually GPU-resident,
         # preventing simultaneous cold launches from exceeding two slots.
         limit=time.time()+180
         while child.poll() is None and child.pid not in resources(gpu)['compute_pids'] and time.time()<limit:time.sleep(1)
        print('START',gpu,slot,job,mode,child.pid,flush=True)
        rc=child.wait();write(RUN/job/(mode+'_exitcode.json'),dict(exitcode=rc,pid=child.pid));assert rc==0,job+' '+mode+' failed; downstream blocked'
       time.sleep(2)
      assert read(RUN/job/'completion.json').get('complete')
      for mode in ['none','mirror4']:assert read(RUN/job/'final_test'/mode/'artifact_audit.json').get('complete') is True
      with mutex:state['completed'].append(job);active.pop((gpu,slot),None);write(RUN/'controller_status.json',state)
     with mutex:state['slots'][key]={'status':'stopped_no_more_admission','updated_unix':time.time()};write(RUN/'controller_status.json',state)
   except BaseException as e:
    failed.set()
    with mutex:state['slots'][key]={'status':'blocked','error':repr(e)};write(RUN/f'failure_gpu{gpu}_slot{slot}.json',dict(error=repr(e),traceback=traceback.format_exc()));write(RUN/'controller_status.json',state)
  threads=[threading.Thread(target=card,args=(g,s)) for g in [0,1] for s in [0,1]]
  for t in threads:t.start()
  for t in threads:t.join()
  state['status']='complete' if len(state['completed'])==12 else 'blocked_or_deadline';write(RUN/'controller_status.json',state)
  assert state['status']=='complete'
if __name__=='__main__':
 try:
  if sys.argv[1]=='prepare':prepare()
  elif sys.argv[1]=='controller':controller()
  else:phase(sys.argv[2],sys.argv[1],int(sys.argv[3]))
 except BaseException as e:
  if RUN.exists():write(RUN/('failure_'+sys.argv[1]+'.json'),dict(error=repr(e),pid=os.getpid(),traceback=traceback.format_exc(),time=time.time()))
  raise

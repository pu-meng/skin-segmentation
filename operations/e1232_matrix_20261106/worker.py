import os,sys,json,time,csv,shutil,subprocess,importlib,copy
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation');sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
import source.e1232_fourarm_screen_20261106.entry as m
from source.e1232_matrix_20261106.candidate import build_candidate,ARMS,ContentB
import torch
RUN=Path('/media/8T/PuMengYu/experiments/skin_segmentation/E1232_GateMatrix_FRESH300_20261106');OPS=ROOT/'operations/e1232_matrix_20261106'
ARM=os.environ.get('MATRIX_ARM','BT');PHASE=sys.argv[1];EVID=OPS/ARM;PATH=RUN/ARM/'train'
m.RUN=RUN;m.UUID='GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b'
def write(p,v):m.write(p,v)
def verify():
 p=m.manifest()
 for f,h in p['source_sha256'].items():assert m.sha256_file(ROOT/f)==h,'source drift '+f
 assert m.sha256_file(p['config']['data']['split_file'])==p['config']['data']['split_sha256']
 assert shutil.disk_usage(RUN).free>8*2**30,'disk below 8GiB'
 return p
m.verify=verify
def barrier(name,timeout=3600):
 until=time.monotonic()+timeout
 cohort=['BU'] if ARM=='BU' else ['BT','BS','AU']
 while not all((OPS/a/name).exists() for a in cohort):
  assert not any((OPS/a/'failure.json').exists() for a in cohort),'peer failed'
  if time.monotonic()>until:raise TimeoutError(name)
  time.sleep(2)
def cpu():
 torch.set_num_threads(2);api=m.runner.baseline_api();records={}
 for arm in ARMS:
  api.seed_all(42);ref=m.build_candidate('R3' if arm=='BS' else 'R1',False);rs={k:v.clone() for k,v in ref.state_dict().items()};ref_parameters=sum(p.numel() for p in ref.parameters());del ref
  api.seed_all(42);model=build_candidate(arm,False);sd=model.state_dict()
  changed=[];matched=0
  for k,v in sd.items():
   if k in rs and v.shape==rs[k].shape:
    if not torch.equal(v,rs[k]):changed.append(k)
    else:matched+=1
  assert not changed,changed
  assert all(k in sd for k in rs if not (arm[-1]=='U' and (k.startswith('decoder.e3.gd.') or any(k.startswith('decoder.e3.d.'+part) for part in ['mix.','n2.','ffn.','post.','gamma'])))),arm
  model.load_state_dict(copy.deepcopy(sd),strict=True)
  if arm[-1]=='U':
   e=model.decoder.e3.eval();s=torch.randn(1,64,64,64,requires_grad=True);d=torch.randn_like(s,requires_grad=True)
   with torch.no_grad():e.gs[-1].weight.fill_(.01)
   out,dd=e(s,d);assert dd is d;out.square().mean().backward()
   assert d.grad is not None and bool(d.grad.abs().max()>0),'D controls S'
   assert all(p.grad is not None for p in e.parameters()),'unused U params'
  b=model.decoder.e2
  if isinstance(b,ContentB):
   x=torch.randn(2,512,8,8);z=torch.cat([b.context(x),b.context_w(x*2)],1)
   for head in [b.alpha,b.beta]:torch.testing.assert_close(head(z),torch.full((2,512,1,1),.1),atol=1e-7,rtol=0)
  records[arm]=dict(parameters=sum(p.numel() for p in model.parameters()),retained_equal_tensors=matched,all_retained_tensors_exact=True,strict_load=True,U_D_exact_identity=arm[-1]=='U',extra_parameters_vs_reference=sum(p.numel() for p in model.parameters())-ref_parameters)
  del model,sd,rs
 assert not torch.cuda.is_initialized()
 write(RUN/'cpu_acceptance.json',dict(passed=True,arms=records,cuda_initialized=False,scope='retained initialization and strict state reload, actual U skip forward/backward, B controller shape/range/init; full model formal batch GPU next'))
 print('CPU_ACCEPTANCE_PASS',records,flush=True)
def training():
 config=verify()['config'];assert json.loads((RUN/'cpu_acceptance.json').read_text())['passed']
 api=m.runner.baseline_api();api.seed_all(42);torch.set_num_threads(4);torch.cuda.set_device(0)
 model=build_candidate(ARM,True).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8)
 scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=50,eta_min=1e-5)
 loaders,generator=m.make_fast_loaders(config,api,workers=2,validation_batch=8);criterion=api.BceDiceLoss().cuda()
 mapping=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
 assert any(m.UUID in l and f', {os.getpid()},' in l for l in mapping.splitlines()),mapping
 PATH.mkdir(parents=True,exist_ok=True)
 write(PATH/'provenance.json',dict(arm=ARM,stable_id=ARMS[ARM],pid=os.getpid(),gpu_uuid=m.UUID,physical_gpu=1,parameters=sum(p.numel() for p in model.parameters()),seed=42,split_sha256=config['data']['split_sha256'],source_sha256=m.manifest()['source_sha256'],imagenet_only=True,initialization='shared original42, Bottleneck45; content new heads46; no trained checkpoint'))
 write(EVID/(PHASE+'_identity.json'),dict(pid=os.getpid(),gpu_uuid=m.UUID,batch=8,precision='FP32',stable_id=ARMS[ARM]))
 try:
  if PHASE=='preflight':
   write(EVID/'resident_ready.json',dict(pid=os.getpid()));barrier('resident_ready.json')
   torch.cuda.reset_peak_memory_stats();model.train();x,y,ids,_=next(iter(loaders['train']))
   opt.zero_grad(set_to_none=True);pred=model(x.cuda());loss=criterion(pred,y.cuda());assert pred.shape==(8,1,256,256);loss.backward();m.finite_gradients(model)
   n,p=next((n,p) for n,p in model.named_parameters() if n.startswith('decoder.e2.alpha.') and p.grad is not None and bool(p.grad.abs().max()>0));before=p.detach().clone();opt.step();torch.cuda.synchronize();assert not torch.equal(before,p)
   write(EVID/'preflight_passed.json',dict(passed=True,pid=os.getpid(),gpu_uuid=m.UUID,real_batch=list(x.shape),loss=loss.item(),parameter=n,numerical_update=True,max_delta=float((before-p).abs().max()),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,cohort=['BU'] if ARM=='BU' else ['BT','BS','AU'],weights_discarded=True))
   print('FORMAL_BATCH8_RESIDENT_PREFLIGHT_PASS',ARM,flush=True);barrier('preflight_passed.json')
  else:
   write(EVID/'formal_ready.json',dict(pid=os.getpid()));barrier('formal_ready.json')
   original_step=opt.step;seen=False
   def step(*a,**kw):
    nonlocal seen
    if seen:return original_step(*a,**kw)
    n,p=next((n,p) for n,p in model.named_parameters() if n.startswith('decoder.e2.alpha.') and p.grad is not None and bool(p.grad.abs().max()>0));before=p.detach().clone();result=original_step(*a,**kw);torch.cuda.synchronize();assert not torch.equal(before,p)
    write(EVID/'first_optimizer_update.json',dict(pid=os.getpid(),gpu_uuid=m.UUID,epoch=1,step=1,optimizer_updates=1,parameter=n,numerical_update=True,max_delta=float((before-p).abs().max()),total_budget=300,loss='see train/status.json'))
    print('FORMAL_OPTIMIZER_UPDATE',ARM,flush=True);seen=True;return result
   opt.step=step
   m.train(ARM,model,opt,scheduler,loaders,generator,criterion,torch.device('cuda:0'),PATH,resume=False,stop=300)
   verify()
 finally:
  for l in loaders.values():
   if hasattr(l,'close'):l.close()
def test():
 verify();import final_test_pipeline as pipeline
 rows=list(csv.DictReader((PATH/'history.csv').open()));assert [int(r['epoch']) for r in rows]==list(range(1,301))
 s=json.loads((PATH/'status.json').read_text());assert s['status']=='budget_complete' and s['epoch_completed']==300 and s['optimizer_updates']==261900
 last=torch.load(PATH/'checkpoints/latest.pth',map_location='cpu',weights_only=False);winner=min(rows,key=lambda r:float(r['val_loss']))
 assert last['epoch']==300 and last['best_epoch']==int(winner['epoch']) and last['best_loss']==float(winner['val_loss']);del last
 out=RUN/ARM/'final_test';out.mkdir(exist_ok=True);snap=out/'fixed_best.pth';assert not snap.exists();cp=PATH/'checkpoints/best.pth'
 h=m.sha256_file(cp);shutil.copy2(cp,snap);assert h==m.sha256_file(cp)==m.sha256_file(snap)
 torch.set_num_threads(4);api=m.runner.baseline_api();api.seed_all(42)
 model=build_candidate(ARM,False);model.load_state_dict(torch.load(snap,map_location='cpu',weights_only=False),strict=True)
 write(out/'cpu_strict_load.json',dict(passed=True,checkpoint_sha256=h,selected_epoch=int(winner['epoch']),history_rows=300,cuda_initialized=torch.cuda.is_initialized()))
 from operations.nnunet_mirror_validation.run_validation import Predictor
 from torch.utils.data import DataLoader
 dataset=api.HAMDataset(Path(verify()['config']['data']['split_file']).parent,'test');ids={r['image_id'] for r in dataset.rows};assert len(ids)==len(dataset)==2020 and not dataset.training
 loader=DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0);torch.cuda.set_device(0);model.cuda().eval();results={}
 for mode in ['none','mirror4']:
  verify();pred=Predictor(model,mode=='mirror4').eval();target=out/mode;assert not target.exists()
  write(out/'status.json',dict(status='testing',mode=mode,pid=os.getpid(),checkpoint_sha256=h))
  result=api.evaluate(pred,loader,api.BceDiceLoss().cuda(),torch.device('cuda:0'),target)
  result.update(arm=ARM,stable_id=ARMS[ARM],mode=mode,split='test',checkpoint_sha256=h,checkpoint=str(snap),selected_epoch=int(winner['epoch']),split_sha256=verify()['config']['data']['split_sha256'],tta=mode=='mirror4',views=[[],[2],[3],[2,3]] if mode=='mirror4' else [[]],selection='earliest validation minimum; test never selects')
  assert pred.verify_remaining==0 and len(pred.times)==2020
  audit=pipeline.validate_outputs(target,ids,result)
  write(target/'summary.json',result);write(target/'artifact_audit.json',audit)
  write(target/'prediction_manifest.json',{p.name:m.sha256_file(p) for p in sorted((target/'predictions').glob('*.png'))})
  (target/'report.txt').write_text('HAM测试2020；固定验证best；'+mode+'；HD95单位为256网格像素。\n'+json.dumps(result,ensure_ascii=False,indent=2))
  pred.close();results[mode]=result
 assert m.sha256_file(snap)==h==m.sha256_file(cp);verify()
 write(out/'results.json',dict(complete=True,results=results));write(EVID/'completion.json',dict(training_complete=True,epochs=300,test_complete=True,test_images=2020,modes=['none','mirror4'],checkpoint_sha256=h));print('TRAIN300_FIXED_BEST_BOTH_TESTS_COMPLETE',ARM,flush=True)
if __name__=='__main__':
 try:
  if PHASE=='cpu':cpu()
  elif PHASE=='test':test()
  else:training()
 except BaseException as e:
  write(EVID/'failure.json',dict(phase=PHASE,pid=os.getpid(),error=repr(e)));raise

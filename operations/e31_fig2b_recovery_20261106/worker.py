"""Isolated operational recovery; original scientific sources unchanged."""
import os,sys,json,csv,hashlib,shutil,inspect,types,random
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
OPS=ROOT/'operations/e31_fig2b_recovery_20261106'
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t');prepare_imports('tinyvim_s')
import torch,numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp.json');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def equal(a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 if isinstance(a,np.ndarray):return np.array_equal(a,b)
 return a==b

def e31(mode):
 os.environ['E31_ARM']='E31-S1'
 import source.e31_skip_positions_20261106.entry
 from source.e31_skip_positions_20261106 import experiment as e
 r=e.runner;r.verify_runtime=e.base.verify_v1_runtime;c=e.prepare();rd=e.RUN/'run';md=rd/'checkpoints'
 manifest=json.loads((OPS/'manifest.json').read_text())['E31-S1']
 assert sha(md/'latest.pth')==manifest['latest_sha256'] and sha(md/'best.pth')==manifest['best_sha256']
 r.verify_runtime=e.base.verify_v1_runtime;e.base.verify_v1_runtime()
 for rel,h in json.loads((e.RUN/'protocol.json').read_text())['source_sha256'].items():assert sha(ROOT/rel)==h,rel
 torch.set_num_threads(4);api=r.baseline_api();api.seed_all(42)
 m=e.build_candidate(pretrained=False);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8)
 sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=50,eta_min=1e-5)
 x=torch.load(md/'latest.pth',map_location='cpu',weights_only=False)
 m.load_state_dict(x['model'],strict=True);opt.load_state_dict(x['optimizer']);sched.load_state_dict(x['scheduler'])
 assert equal(m.state_dict(),x['model']) and equal(opt.state_dict(),x['optimizer']) and equal(sched.state_dict(),x['scheduler'])
 rows=list(csv.DictReader((rd/'history.csv').open()));epoch=x['epoch'];updates=epoch*c['training']['updates_per_epoch']
 assert [int(v['epoch']) for v in rows]==list(range(1,epoch+1))
 assert int(rows[-1]['optimizer_updates'])==updates and all(int(v['step'])==updates for v in opt.state.values())
 best=min(rows,key=lambda v:float(v['val_loss']))
 assert x['best_epoch']==int(best['epoch']) and x['best_loss']==float(best['val_loss'])
 assert sched.last_epoch==epoch and opt.param_groups[0]['lr']==sched.get_last_lr()[0]
 assert x['protocol_sha256']==sha(e.RUN/'protocol.json') and x['split_sha256']==c['data']['split_sha256']
 loaders,g=e.make_fast_loaders(c,api,workers=2,validation_batch=8);g.set_state(x['loader_rng']);assert equal(g.get_state(),x['loader_rng'])
 random.setstate(x['python_rng']);np.random.set_state(x['numpy_rng']);torch.set_rng_state(x['torch_rng'])
 assert equal(random.getstate(),x['python_rng']) and equal(np.random.get_state(),x['numpy_rng']) and equal(torch.get_rng_state(),x['torch_rng'])
 bm=e.build_candidate(pretrained=False);bm.load_state_dict(torch.load(md/'best.pth',map_location='cpu',weights_only=False),strict=True);del bm
 audit=dict(status='passed',checkpoint=str(md/'latest.pth'),checkpoint_sha256=manifest['latest_sha256'],completed_epoch=epoch,resume_epoch=epoch+1,optimizer_updates=updates,model_strict_exact=True,optimizer_exact=True,scheduler_exact=True,best_selection_exact=True,loader_rng_exact=True,global_rng_exact=True,worker_rng='PlannedDataset deterministic from complete parent-generated plans; worker_init sets threads only; bitwise replay not promised',cuda_rng_valid=x['cuda_rng'].dtype==torch.uint8 and x['cuda_rng'].numel()>0,CUDA_initialized=torch.cuda.is_initialized())
 assert audit['cuda_rng_valid'] and not audit['CUDA_initialized'];save(OPS/'E31-S1_checkpoint_audit.json',audit);print('CPU_RECOVERY_AUDIT',json.dumps(audit),flush=True)
 if mode=='audit':return
 torch.cuda.set_device(0);device=torch.device('cuda:0');m.to(device)
 # Original runner performs complete state restoration again immediately before next forward.
 source=inspect.getsource(r.train)
 old="        from recover_ham_dedup import inspect_run\n        record = inspect_run(provenance['model'], config)"
 assert source.count(old)==1;source=source.replace(old,'        record = recovery_audit')
 needle='    for epoch in range(start_epoch, epochs + 1):\n'
 assert source.count(needle)==1;source=source.replace(needle,needle+"        assert shutil.disk_usage(ROOT).free >= 8*2**30 + 220*2**20, 'Disk reserve before next full epoch; saved boundary, no cleanup'\n")
 ns=dict(r.train.__globals__,recovery_audit=audit,shutil=shutil,ROOT=ROOT);exec(compile(source,str(OPS/'isolated_e31_train.py'),'exec'),ns)
 info=json.loads((rd/'provenance.json').read_text());info.update(pid=os.getpid(),fresh_start=False,recovery_operations=str(OPS))
 save(OPS/'E31-S1_resume_provenance.json',info)
 try:ns['train'](m,loaders,api.BceDiceLoss().to(device),opt,device,rd,c,info,g,resume=True)
 finally:
  for l in loaders.values():
   if hasattr(l,'close'):l.close()

def fig(mode):
 SRC=ROOT/'source/mambalite_publiccode_four_seeds_20261106'
 sys.path[:0]=[str(SRC),str(ROOT/'source/mambalite_paper_candidate_20261105')]
 import reconstruction_engine as re
 import reconstruction_restore as restore
 manifest=json.loads((OPS/'manifest.json').read_text())['fig2b42']
 src=inspect.getsource(restore.restore_cpu)
 for old,new in [("c['reconstruction_resume']['input_latest_sha256']","recovery_identity['latest_sha256']"),("c['reconstruction_resume']['input_best_sha256']","recovery_identity['best_sha256']")]:
  assert src.count(old)==1;src=src.replace(old,new)
 ns=dict(restore.__dict__,recovery_identity=manifest);exec(compile(src,str(OPS/'isolated_fig_restore.py'),'exec'),ns);restore.restore_cpu=ns['restore_cpu']
 torch.set_num_threads(4);re.verify_sources();re.api.seed_all(42);m=re.build_model();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8);sc=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=50,eta_min=1e-5)
 loaders=re.make_loaders(42,workers=0);payload,audit=restore.restore_cpu(m,opt,sc,loaders,42,re.run_path(42),re.model_path(42))
 bm=re.build_model();bm.load_state_dict(torch.load(re.model_path(42)/'best.pth',map_location='cpu',weights_only=False),strict=True)
 assert not torch.cuda.is_initialized();audit['status']='passed';save(OPS/'fig2b42_checkpoint_audit.json',audit);print('CPU_RECOVERY_AUDIT',json.dumps(audit),flush=True)
 if mode=='audit':return
 del m,opt,sc,bm,loaders,payload
 source=inspect.getsource(re.train);needle=' for epoch in range(start_epoch,301):\n';assert source.count(needle)==1
 source=source.replace(needle,needle+"  assert shutil.disk_usage(ROOT).free >= 8*2**30 + 32*2**20, 'Disk reserve before next full epoch; saved boundary, no cleanup'\n")
 ns=dict(re.__dict__,shutil=shutil);exec(compile(source,str(OPS/'isolated_fig_train.py'),'exec'),ns);ns['train'](42)

def fresh(arm):
 if arm=='E31-S2':
  os.environ['E31_ARM']=arm
  import source.e31_skip_positions_20261106.entry
  from source.e31_skip_positions_20261106 import experiment as e
 else:
  assert arm=='DG-S0'
  import source.direct_skip_gate_20261106.entry
  from source.direct_skip_gate_20261106 import experiment as e
 r=e.runner;source=inspect.getsource(r.train);needle='    for epoch in range(start_epoch, epochs + 1):\n';assert source.count(needle)==1
 source=source.replace(needle,needle+"        assert shutil.disk_usage(ROOT).free >= 8*2**30 + 220*2**20, 'Disk reserve before next full epoch; saved boundary, no cleanup'\n")
 ns=dict(r.train.__globals__,shutil=shutil,ROOT=ROOT);exec(compile(source,str(OPS/'isolated_fresh_train.py'),'exec'),ns);r.train=ns['train'];e.main('train')

if __name__=='__main__':
 arm,mode=sys.argv[1:];assert arm in ['E31-S1','fig2b42','E31-S2','DG-S0'] and mode in ['audit','train','fresh']
 try:
  if mode=='fresh':fresh(arm)
  else:(e31 if arm=='E31-S1' else fig)(mode)
 except BaseException as exc:
  import traceback
  save(OPS/(arm+'_'+mode+'_failure.json'),dict(error=repr(exc),traceback=traceback.format_exc(),pid=os.getpid()));raise

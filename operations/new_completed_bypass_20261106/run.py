import os,sys,json,csv,time,hashlib,shutil,fcntl,signal,traceback,subprocess
from pathlib import Path
R=Path('/home/PuMengYu/experiments/skin_segmentation')
OPS=R/'operations/new_completed_bypass_20261106'
O=Path('/media/8T/PuMengYu/diagnostics/new_completed_bypass_20261106')
os.environ['BD_ARM']='D4-B3D'
sys.path[:0]=[str(R),str(R/'source/comparison'),str(R/'source/bottleneck_mamba'),str(R/'operations/nnunet_mirror_validation')]
import numpy as np
import torch
from torch.utils.data import DataLoader
import ham_fast_runner as existing
api=existing.baseline_api()
from project_paths import PATHS
from source.b3_e1_e31_20261105.candidate import build_candidate as build_e
from source.bd_controls_20261105.candidate import build_candidate as build_d
from source.bd_controls_20261105.official_mednext import MedNeXtBlock
from run_validation import Predictor
ARMS={'B3-E1-E3.1':('TinyViM_B3_E1_E3.1_HAM_20261105',67,['full','B3_off','E1_WPO_off','E31_off']), 'D4-B3D-R2':('TinyViM_B3D原配置重跑_HAM_20261106',129,['full','B3_off','Med_residual_off_keep_projection'])}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');q.replace(p)
def build(a):return build_e(pretrained=False) if a=='B3-E1-E3.1' else build_d(pretrained=False,count=3,med=True)
def sourcefiles():
 files={Path(__file__),Path(PATHS['active_split']),R/'operations/nnunet_mirror_validation/run_validation.py',R/'operations/nnunet_mirror_validation/mirror_core.py'}
 for d in ['source/b3_e1_e31_20261105','source/bd_controls_20261105','Models/E3DynamicSkip','Models/E2Simplified','source/three_module_e23_20261104','source/three_module_impl_20261104','source/bottleneck_mamba','source/comparison','Models/TinyViM']:
  files.update((R/d).rglob('*.py'))
 return {str(p):sha(p) for p in sorted(files)}
def statehash(m):
 h=hashlib.sha256()
 for n,v in m.state_dict().items():h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def prepare():
 assert not O.exists(),'Refuse overwrite'
 O.mkdir(parents=True)
 rec={}
 for a,(root,epoch,groups) in ARMS.items():
  arm=R/'log'/root/a;s=arm/'final_test'/a/'test/summary.json';summary=json.loads(s.read_text());ck=Path(summary['checkpoint']);before=sha(ck)
  assert before==summary['checkpoint_sha256'] and summary['selected_epoch']==epoch
  with (arm/'run/history.csv').open() as f:hist=list(csv.DictReader(f))
  assert len(hist)==300 and int(min(hist,key=lambda r:float(r['val_loss']))['epoch'])==epoch
  dest=O/a/'fixed_best.pth';dest.parent.mkdir();shutil.copyfile(ck,dest)
  assert sha(ck)==sha(dest)==before
  m=build(a);msg=m.load_state_dict(torch.load(dest,map_location='cpu'),strict=True)
  assert not torch.cuda.is_initialized()
  if a=='D4-B3D-R2':
   blocks=[v for v in m.modules() if type(v) is MedNeXtBlock];assert len(blocks)==8 and all(v.do_res and v.dim=='2d' for v in blocks)
   assert len(m.decoder.enhance)==3
  rec[a]={'original_checkpoint':str(ck),'snapshot':str(dest),'checkpoint_sha256':before,'selected_epoch':epoch,'strict_cpu_load':str(msg),'history_rows':len(hist),'summary':str(s)}
  del m
 before=sourcefiles()
 write(O/'protocol.json',{'scope':'fixed best val-only inference diagnostic, no training/checkpoint reselection/test configuration selection','images':1010,'batch':1,'workers':0,'precision':'FP32','split':str(PATHS['active_split']),'split_sha256':sha(PATHS['active_split']),'groups':{a:v[2] for a,v in ARMS.items()},'modes':['none','mirror4'],'views':[[],[2],[3],[2,3]],'fusion':'inverse flipped real head logits arithmetic mean then sigmoid, threshold >=0.5','E1_WPO_off':'DStage output original X; closes WPO correction/removes replacement layer, NOT restoration of original DoubleConv','E31_off':'independent eval forward hook returns original S, leaves D unchanged','Med_residual_off_keep_projection':'all eight official equal-width do_res MedNeXtBlock outputs return original inputs; retain every stage channel projection, ups, head, skips; NOT removal of decoder or restoration of original DoubleConv','B3_off':'entire enhance sequence output original X','GPU':1,'shared_authorized':True,'protected_processes':'never stop or modify other jobs','deadline_seconds':7200,'OOM':'fail blocked; no CPU fallback/condition change','sources':before,'checkpoints':rec,'torch_cuda_initialized':torch.cuda.is_initialized()})
 write(O/'cpu_preflight.json',{'passed':True,'models':rec});print('CPU_STRICT_PASS',flush=True)
def off(m,g):
 mods=[]
 if g=='B3_off':mods=[m.decoder.enhance]
 elif g=='E1_WPO_off':mods=[m.decoder.stages[3]]
 elif g=='E31_off':mods=[m.decoder.refiner]
 elif g=='Med_residual_off_keep_projection':
  mods=[v for v in m.modules() if type(v) is MedNeXtBlock];assert len(mods)==8 and all(v.do_res for v in mods)
 return [v.register_forward_hook(lambda module,inputs,output:inputs[0]) for v in mods]
def run():
 signal.signal(signal.SIGALRM,lambda a,b:(_ for _ in ()).throw(TimeoutError('7200s deadline')));signal.alarm(7200)
 p=json.loads((O/'protocol.json').read_text());assert sourcefiles()==p['sources']
 torch.set_num_threads(4);api.seed_all(42)
 data=api.HAMDataset(Path(PATHS['active_split']).parent,'val');assert len(data)==1010 and not data.training
 ids=[r['image_id'] for r in data.rows];assert len(set(ids))==1010
 write(O/'validation_identity.json',{'ids':ids,'split_sha256':sha(PATHS['active_split']),'pixel_access':'only val row indices; shared mmap contains other splits but none indexed'})
 loader=DataLoader(data,batch_size=1,num_workers=0,shuffle=False)
 torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.18,0)
 write(O/'gpu_before.json',{'pid':os.getpid(),'physical_gpu':1,'uuid':'GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b','snapshot':subprocess.check_output(['nvidia-smi'],text=True)})
 results={};checks={}
 for a,(_,epoch,groups) in ARMS.items():
  m=build(a);m.load_state_dict(torch.load(p['checkpoints'][a]['snapshot'],map_location='cpu'),strict=True);m.eval().requires_grad_(False).cuda()
  state0=statehash(m);buffers={n:v.detach().cpu().clone() for n,v in m.named_buffers()};x=data[0][0].unsqueeze(0).cuda()
  with torch.no_grad():
   full=m(x).clone();assert bool(torch.isfinite(full).all())
   for g in groups[1:]:
    handles=off(m,g);v=m(x);assert v.shape==full.shape and bool(torch.isfinite(v).all())
    for h in handles:h.remove()
    assert torch.equal(m(x),full),'Restore full exact'
  checks[a]={'first_val_image':ids[0],'full_restoration_exact':True,'all_interventions_finite':True,'state_hash':state0}
  write(O/'gpu_preflight.json',{'passed':True,'checks':checks,'pid':os.getpid()});print('GPU_PREFLIGHT_PASS',a,flush=True)
  results[a]={}
  for mode in ['none','mirror4']:
   for g in groups:
    target=O/a/mode/g;assert not target.exists()
    write(O/'status.json',{'status':'evaluating','arm':a,'mode':mode,'group':g,'pid':os.getpid(),'updated':time.time()})
    handles=off(m,g);pred=Predictor(m,mode=='mirror4').eval()
    with torch.no_grad():
     for _ in range(4):pred(x)
    assert pred.verify_remaining==0;pred.times.clear()
    print('START',a,mode,g,flush=True)
    r=api.evaluate(pred,loader,api.BceDiceLoss().cuda(),torch.device('cuda'),target)
    assert len(pred.times)==1010
    r.update(arm=a,group=g,mode=mode,split='val',selected_epoch=epoch,checkpoint_sha256=p['checkpoints'][a]['checkpoint_sha256'],split_sha256=p['split_sha256'],logits_sigmoid_max_error=pred.max_error,forward_seconds=sum(pred.times))
    with (target/'per_image_metrics.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==len({v['image_id'] for v in rows})==1010 and {v['image_id'] for v in rows}==set(ids)
    assert {v.name for v in (target/'predictions').glob('*.png')}=={i+'_segmentation.png' for i in ids}
    for k in ['tp','fp','fn','tn']:assert sum(int(v[k]) for v in rows)==r['global'][k]
    write(target/'summary.json',r);results[a][mode+'/'+g]=r;write(O/'partial_results.json',results)
    pred.close()
    for h in handles:h.remove()
    assert statehash(m)==state0 and all(torch.equal(v.detach().cpu(),buffers[n]) for n,v in m.named_buffers())
    print('COMPLETE',a,mode,g,r['global']['dice'],flush=True)
  del pred,m,x;torch.cuda.empty_cache()
 assert sourcefiles()==p['sources']
 for v in p['checkpoints'].values():assert sha(v['snapshot'])==sha(v['original_checkpoint'])==v['checkpoint_sha256']
 write(O/'results.json',{'complete':True,'results':results,'groups_completed':sum(len(v) for v in results.values()),'sources_unchanged':True,'state_and_buffers_unchanged':True})
 write(O/'status.json',{'status':'inference_complete_pending_pixel_audit','pid':os.getpid(),'groups_completed':sum(len(v) for v in results.values())})
if __name__=='__main__':
 try:
  if sys.argv[1]=='prepare':prepare()
  else:
   with (O/'RUNNING.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (O/'status.json').exists();(O/'pid').write_text(str(os.getpid()));run()
 except BaseException as e:
  if O.exists():write(O/'failure.json',{'error':repr(e),'traceback':traceback.format_exc()});write(O/'status.json',{'status':'failed_blocked','error':repr(e)})
  raise

import os,sys,json,hashlib,time,csv,subprocess,traceback,fcntl
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
OPS=ROOT/'operations/nnunet_mirror_validation'
OUT=ROOT/'log/nnunet_mirror_validation_20261105'
sys.path[:0]=[str(OPS),str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba'),str(ROOT/'source/three_module_impl_20261104')]
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import ham_fast_runner as existing
from project_paths import PATHS
from mirror_core import mirror_logits
api=existing.baseline_api()
from final_test_four_stage_decoder import build_four_stage_baseline
from candidate import build_candidate
RUNS={'E0':ROOT/'log/TinyViM_四层对应Decoder_基础_HAM_20261104/run','E1':ROOT/'log/TinyViM_三模块_E1_DWPO_HAM_20261104/run'}
EXPECTED={'E0':'fa93b6abcfda2777a4da7021b6fd9f7335aa85113ae4d8b172bf72ac7d17a34f','E1':'4a8be539bbad6d51dd15442f090740fdde7a063f8fac04b29f58c9fe0f476c9b'}
SPLIT_SHA='0f5e2df296f50afb74002ce94438a5b7a17782deaef671f93673a60a8e8b5ed8'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def build(arm):return build_four_stage_baseline(pretrained=False) if arm=='E0' else build_candidate(pretrained=False,d=True)
def identities():
 files=set()
 for arm,run in RUNS.items():
  prov=json.loads((run/'provenance.json').read_text())
  files.update(ROOT/p for p in prov['source_sha256'])
  files.update([run/'checkpoints/best.pth',run/'validation/summary.json',run/'provenance.json',run/'history.csv'])
 files.add(Path(PATHS['active_split']))
 files.add(Path('/home/PuMengYu/nnUNet/nnunetv2/inference/predict_from_raw_data.py'))
 return {str(p):sha(p) for p in sorted(files)}
def prepare():
 assert not OUT.exists(),'Refuse existing output root'
 OUT.mkdir()
 records={}
 for arm,run in RUNS.items():
  old=json.loads((run/'validation/summary.json').read_text());audit=json.loads((run/'artifact_audit.json').read_text());prov=json.loads((run/'provenance.json').read_text())
  assert audit['training_complete'] and audit['epochs']==300
  assert old['split']=='val' and old['images']==1010 and old['split_sha256']==SPLIT_SHA
  assert old['checkpoint_sha256']==sha(run/'checkpoints/best.pth')==EXPECTED[arm]
  with (run/'history.csv').open() as f:history=list(csv.DictReader(f))
  assert len(history)==300
  best=min(history,key=lambda r:float(r['val_loss']))
  assert int(best['epoch'])==old['selected_epoch']
  for rel in ['Models/TinyViM/four_stage_decoder.py','Models/TinyViM/model.py']+(['source/three_module_impl_20261104/candidate.py','source/three_module_impl_20261104/official_wave.py'] if arm=='E1' else []):
   assert sha(ROOT/rel)==prov['source_sha256'][rel],rel
  model=build(arm);state=torch.load(run/'checkpoints/best.pth',map_location='cpu');load=model.load_state_dict(state,strict=True)
  records[arm]={'strict_load':str(load),'selected_epoch':old['selected_epoch'],'checkpoint_sha256':EXPECTED[arm],'old_summary':str(run/'validation/summary.json'),'selection':'minimum validation loss from original 300-row history, fixed existing best; no reselection'}
  del model,state
 assert sha(PATHS['active_split'])==SPLIT_SHA
 assert not torch.cuda.is_initialized()
 write(OUT/'cpu_preflight.json',{'passed':True,'cuda_initialized':False,'models':records,'hashes_before':identities()})
 print('CPU STRICT LOAD PASSED',json.dumps(records),flush=True)
class Predictor(torch.nn.Module):
 def __init__(self,model,tta):
  super().__init__();self.model=model;self.tta=tta;self.times=[];self.verify_remaining=4;self.max_error=0.;self.raw=None
  self.handle=model.decoder.head.register_forward_hook(self.capture)
 def capture(self,m,i,o):self.raw=o
 def logits(self,x):
  self.raw=None;prob=self.model(x)
  assert self.raw is not None
  logit=F.interpolate(self.raw,size=(256,256),mode='bilinear',align_corners=False)
  if self.verify_remaining:
   err=float((torch.sigmoid(logit)-prob).abs().max());self.max_error=max(self.max_error,err)
   torch.testing.assert_close(torch.sigmoid(logit),prob,rtol=0,atol=0);self.verify_remaining-=1
  return logit
 def forward(self,x):
  torch.cuda.synchronize();t=time.perf_counter()
  p=torch.sigmoid(mirror_logits(self.logits,x,self.tta))
  torch.cuda.synchronize();self.times.append(time.perf_counter()-t)
  if len(self.times)%100==0:print('FORWARD',len(self.times),'mirror4' if self.tta else 'none',flush=True)
  return p
 def close(self):self.handle.remove()
def validate_artifacts(path,ids,result):
 with (path/'per_image_metrics.csv').open() as f:rows=list(csv.DictReader(f))
 assert len(rows)==len(ids)==1010 and {r['image_id'] for r in rows}==ids
 assert {p.name for p in (path/'predictions').glob('*.png')}=={i+'_segmentation.png' for i in ids}
 for k in ['tp','fp','fn','tn']:assert sum(int(r[k]) for r in rows)==result['global'][k]
 for k in ['dice','iou']:assert abs(float(np.mean([float(r[k]) for r in rows]))-result['per_image_mean'][k])<1e-12
 return {'images':len(rows),'predictions':1010,'per_image_csv_sha256':sha(path/'per_image_metrics.csv'),'prediction_manifest_sha256':sha(path/'prediction_manifest.json')}
def run():
 pre=json.loads((OUT/'cpu_preflight.json').read_text());assert pre['passed'];assert identities()==pre['hashes_before']
 torch.set_num_threads(4);api.seed_all(42);torch.cuda.set_device(0)
 torch.cuda.set_per_process_memory_fraction(.25,0)
 data=api.HAMDataset(Path(PATHS['active_split']).parent,'val');assert len(data)==1010 and not data.training
 ids={r['image_id'] for r in data.rows};assert len(ids)==1010
 loader=DataLoader(data,batch_size=1,num_workers=0,shuffle=False)
 write(OUT/'gpu_before.json',{'gpu':0,'snapshot':subprocess.check_output(['nvidia-smi'],text=True),'shared_gpu_authorized':True,'protected_existing_processes':'no terminate/modify','batch_size':1,'precision':'FP32','allocator_budget_fraction':.25})
 all_results={}
 for arm,oldrun in RUNS.items():
  model=build(arm);model.load_state_dict(torch.load(oldrun/'checkpoints/best.pth',map_location='cpu'),strict=True);model.to('cuda').eval()
  all_results[arm]={}
  for mode in ['none','mirror4']:
   write(OUT/'status.json',{'status':'evaluating','arm':arm,'mode':mode,'pid':os.getpid(),'test_data_accessed':False})
   pred=Predictor(model,mode=='mirror4').eval();x=data[0][0].unsqueeze(0).cuda()
   with torch.no_grad():
    for _ in range(5):pred(x)
   assert pred.verify_remaining==0
   pred.times.clear();torch.cuda.reset_peak_memory_stats()
   path=OUT/arm/mode
   print('START',arm,mode,flush=True)
   result=api.evaluate(pred,loader,api.BceDiceLoss().cuda(),torch.device('cuda'),path)
   assert len(pred.times)==1010
   result.update(arm=arm,mode=mode,split='val',test_data_accessed=False,checkpoint_sha256=EXPECTED[arm],selected_epoch=pre['models'][arm]['selected_epoch'],split_sha256=SPLIT_SHA,logits_probability_max_abs_error=pred.max_error)
   result['latency']={'mean_ms':float(np.mean(pred.times)*1000),'median_ms':float(np.median(pred.times)*1000),'p95_ms':float(np.percentile(pred.times,95)*1000),'total_seconds':sum(pred.times),'images':1010,'warmup_forwards':5,'boundary':'GPU-resident input; synchronized wrapper forward incl mirror flips, original network sigmoid, head-logit resize, fusion and output sigmoid; excludes H2D, loss, metrics, I/O; shared concurrent GPU load, not isolated benchmark','peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20}
   with (path/'latency.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['image_id','forward_seconds']);w.writerows(zip([r['image_id'] for r in data.rows],pred.times))
   write(path/'prediction_manifest.json',{p.name:sha(p) for p in sorted((path/'predictions').glob('*.png'))})
   result['artifact_audit']=validate_artifacts(path,ids,result)
   if mode=='none':
    old=json.loads((oldrun/'validation/summary.json').read_text())
    deltas={k:result['global'][k]-old['global'][k] for k in result['global']}
    result['old_validation_delta']=deltas
    write(path/'baseline_comparison.json',deltas)
    assert all(v==0 for v in deltas.values()),f'No-TTA mismatch {deltas}; block comparison'
    with (oldrun/'validation/per_image_metrics.csv').open() as f:oldrows={r['image_id']:r for r in csv.DictReader(f)}
    with (path/'per_image_metrics.csv').open() as f:newrows=list(csv.DictReader(f))
    assert all(all(float(r[k])==float(oldrows[r['image_id']][k]) for k in ['dice','iou','tp','fp','fn','tn']) for r in newrows),'Per-image baseline mismatch'
   write(path/'summary.json',result);all_results[arm][mode]=result
   print('COMPLETE',arm,mode,json.dumps({'global':result['global'],'per_image_mean':result['per_image_mean'],'latency':result['latency']}),flush=True)
   pred.close();del pred
  del model;x=None;torch.cuda.empty_cache()
 after=identities();assert after==pre['hashes_before'],'Protected source/checkpoint changed'
 write(OUT/'hashes_after.json',after)
 write(OUT/'results.json',{'complete':True,'results':all_results,'protocol':{'views':[[],[2],[3],[2,3]],'fusion':'inverse-flipped real pre-sigmoid logits arithmetic mean then sigmoid, threshold >=0.5','input':'256x256 whole image; no sliding windows or Gaussian','precision':'FP32 eval no_grad','test_data_accessed':False,'baseline_check':'exact historical global counts/metrics and per-image Dice/IoU/counts','timing_caveat':'one pass; none then mirror4; shared GPU concurrent training; not isolated speed benchmark'}})
 write(OUT/'gpu_after.json',{'snapshot':subprocess.check_output(['nvidia-smi'],text=True)})
 write(OUT/'status.json',{'status':'complete','pid':os.getpid(),'test_data_accessed':False})
if __name__=='__main__':
 try:
  if sys.argv[1]=='prepare':prepare()
  else:
   with (OUT/'RUNNING.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run()
 except BaseException as e:
  if OUT.exists():write(OUT/'status.json',{'status':'failed','error':repr(e),'traceback':traceback.format_exc(),'pid':os.getpid()})
  raise

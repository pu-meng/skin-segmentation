"""Isolated fixed-best four-view completion; no training or shared writes."""
import os,sys,json,csv,time,hashlib,importlib,traceback,signal
from pathlib import Path
R=Path('/home/PuMengYu/experiments/skin_segmentation');OPS=R/'operations/ham_comparison_tta_20261108';OUT=Path('/media/8T/PuMengYu/experiments/skin_segmentation/HAM_comparison_TTA_20261108')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def main(ident):
 manifest=json.loads((OPS/'manifest.json').read_text());manifest['gpu_uuid']=['GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde','GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b'][int(os.environ['TTA_PHYSICAL_GPU'])];e=next(e for e in manifest['rows'] if e['id']==ident);work=OUT/ident;status=work/'status.json'
 def state(s,**kw):save(status,dict(id=ident,name=e['name'],status=s,pid=os.getpid(),gpu=int(os.environ['TTA_PHYSICAL_GPU']),gpu_uuid=manifest['gpu_uuid'],updated_unix=time.time(),**kw))
 signal.signal(signal.SIGALRM,lambda *a:(_ for _ in ()).throw(TimeoutError('per-arm 4h deadline')));signal.alarm(4*3600)
 os.environ.update(e['env']);os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_THREADING_LAYER='GNU')
 sys.path[:0]=[str(R/p) for p in e['import_prefix']]+[str(R),str(R/'source/comparison'),str(R/'source/bottleneck_mamba')]
 import numpy as np
 import torch
 from PIL import Image
 from torch.nn import functional as F
 from torch.utils.data import DataLoader
 if e.get('api_module')=='engine':api=importlib.import_module('engine').baseline_api()
 else:
  from pretrained_unet2d import prepare_imports
  prepare_imports('mambavision_t');prepare_imports('tinyvim_s')
  import ham_fast_runner
  api=ham_fast_runner.baseline_api()
 torch.set_num_threads(2);api.seed_all(e['reference'].get('seed',42));state('cpu_strict_loading')
 mod=importlib.import_module(e['builder_module']);model=getattr(mod,e['builder_function'])(**e['builder_kwargs']);model.load_state_dict(torch.load(e['checkpoint'],map_location='cpu',weights_only=False),strict=True);assert not torch.cuda.is_initialized()
 assert sha(e['checkpoint'])==e['checkpoint_sha256'];assert sha(e['split_file'])==e['split_sha256'];assert sha(e['original_summary'])==e['original_summary_sha256']
 protected={str(p):sha(p) for p in [Path(e['checkpoint']),Path(e['split_file']),Path(e['original_summary'])]}
 for m in list(sys.modules.values()):
  f=getattr(m,'__file__',None)
  if f and str(f).startswith(str(R)) and f.endswith('.py') and '/operations/all_missing_tta_' not in f:protected[f]=sha(f)
 save(work/'cpu_strict_load.json',dict(passed=True,cuda_initialized=False,builder=e['builder_module']+'.'+e['builder_function'],kwargs=e['builder_kwargs'],checkpoint=e['checkpoint'],checkpoint_sha256=e['checkpoint_sha256'],epoch=e['epoch'],protected_hashes=protected))
 if e['dataset']=='ISIC2018':
  from source.isic2018.data import ISICDataset,CACHE
  ds=ISICDataset(CACHE,'test')
 else:ds=api.HAMDataset(Path(e['split_file']).parent,'test')
 ids=json.loads((work/'test_ids.json').read_text());assert len(ds)==len(ids)==e['test_unique_ids'] and {r['image_id'] for r in ds.rows}==set(ids) and not ds.training
 save(work/'dataset_identity.json',dict(split_file=e['split_file'],split_sha256=e['split_sha256'],dataset=e['dataset'],ids=ids,unique_count=len(ids),row_order=[r['image_id'] for r in ds.rows]))
 os.environ['CUDA_VISIBLE_DEVICES']=manifest['gpu_uuid'];torch.cuda.set_device(0);assert torch.cuda.is_available();model.cuda().eval().requires_grad_(False)
 # Reuse the existing formal four-view Predictor and mirror_logits; select only
 # architecture-specific real head restoration, validated against actual forward.
 sys.path.insert(0,str(R/'operations/nnunet_mirror_validation'))
 from mirror_core import mirror_logits
 class Predictor(torch.nn.Module):
  def capture(self,m,i,o):self.raw=o
  def close(self):self.handle.remove()
  def __init__(self,m):
   torch.nn.Module.__init__(self);self.model=m;self.raw=None;self.times=[];self.verify_remaining=4;self.max_error=0.;self.n=0
   head=m.final if e['logit_mode']=='mambalite' else m.network if e['logit_mode']=='surmt' else m.head if e['logit_mode']=='unet' else m.decoder.head
   self.handle=head.register_forward_hook(self.capture)
  def logits(self,x):
   self.raw=None;p=self.model(x);assert self.raw is not None
   z=F.interpolate(self.raw,scale_factor=2,mode='bilinear',align_corners=True) if e['logit_mode']=='mambalite' else F.interpolate(self.raw,size=(256,256),mode='bilinear',align_corners=False) if e['logit_mode']=='four_stage' else self.raw
   if self.verify_remaining:
    torch.testing.assert_close(z.sigmoid(),p,rtol=0,atol=0);self.max_error=max(self.max_error,float((z.sigmoid()-p).abs().max()));self.verify_remaining-=1
   return z
  def forward(self,x):
   t=time.monotonic();p=mirror_logits(self.logits,x,True).sigmoid();assert bool(torch.isfinite(p).all());self.times.append(time.monotonic()-t);self.n+=1
   if self.n==1 or self.n%50==0:state('testing',completed_images=self.n,total_images=len(ds));print('TTA_FORWARD',ident,self.n,'/',len(ds),flush=True)
   return p
 pred=Predictor(model).eval();errors=[]
 with torch.no_grad():
  for i in range(3):
   x=ds[i][0].unsqueeze(0).cuda();p=pred.logits(x).sigmoid();old=np.asarray(Image.open(Path(e['original_summary']).parent/'predictions'/(ds.rows[i]['image_id']+'_segmentation.png')))>0
   assert np.array_equal(old,p[0,0].cpu().numpy()>=.5),'original noTTA PNG mismatch';errors.append(dict(id=ds.rows[i]['image_id'],mask_mismatch_pixels=0))
 import subprocess
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True);assert any(l.split(',')[0].strip()==manifest['gpu_uuid'] and l.split(',')[1].strip()==str(os.getpid()) for l in apps.splitlines())
 save(work/'gpu_preflight.json',dict(passed=True,pid=os.getpid(),gpu_uuid=manifest['gpu_uuid'],noTTA_exact_PNG=errors,hook=e['logit_mode'],real_forward_sigmoid_exact=True,compute_apps=apps));print('GPU_PREFLIGHT_PASSED',ident,os.getpid(),flush=True)
 target=Path(e['output']);assert not target.exists(),'no overwrite';state('testing',completed_images=0,total_images=len(ds));start=time.monotonic()
 result=api.evaluate(pred,DataLoader(ds,batch_size=1,shuffle=False,num_workers=0),api.BceDiceLoss().cuda(),torch.device('cuda:0'),target)
 result.update(test_used_for_checkpoint_selection=False,physical_gpu=int(os.environ['TTA_PHYSICAL_GPU']),dataset=e['dataset'],split='test',seed=e['reference'].get('seed',42),selected_epoch=e['epoch'],checkpoint=e['checkpoint'],checkpoint_sha256=e['checkpoint_sha256'],split_sha256=e['split_sha256'],stable_id=ident,tta=True,mode='mirror4',views=[[],[2],[3],[2,3]],fusion='inverse-flipped real logits arithmetic mean, then sigmoid',threshold=.5,precision='FP32',batch_size=1,no_tta_summary=e['original_summary'],builder=e['builder_module']+'.'+e['builder_function'],elapsed_seconds=time.monotonic()-start)
 save(target/'summary.json',result);assert pred.n==len(ds)
 state('auditing',completed_images=len(ds),total_images=len(ds))
 records=list(csv.DictReader((target/'per_image_metrics.csv').open()));by={r['image_id']:r for r in records};assert len(records)==len(by)==len(ids) and set(by)==set(ids)
 assert {p.name for p in (target/'predictions').glob('*.png')}=={i+'_segmentation.png' for i in ids};assert len(list((target/'visualizations').glob('*.png')))==12
 totals=np.zeros(4,dtype=np.int64);hds=[];pngs={}
 for row in ds.rows:
  i=row['image_id'];png=target/'predictions'/(i+'_segmentation.png');p=np.asarray(Image.open(png))>0;y=ds.masks[int(row['cache_index'])]>0;cnt=api.confusion(p,y);totals+=cnt
  assert all(int(by[i][k])==int(v) for k,v in zip(['tp','fp','fn','tn'],cnt));hd=api.hd95(p,y);hds.append(hd);assert abs(hd-float(by[i]['hd95_pixels']))<1e-9;pngs[png.name]=sha(png)
 assert all(int(result['global'][k])==int(v) for k,v in zip(['tp','fp','fn','tn'],totals))
 tp,fp,fn,tn=map(int,totals);metrics=dict(dice=2*tp/(2*tp+fp+fn),iou=tp/(tp+fp+fn),accuracy=(tp+tn)/(tp+fp+fn+tn),sensitivity=tp/(tp+fn),specificity=tn/(tn+fp))
 assert all(abs(result['global'][k]-v)<1e-12 for k,v in metrics.items());assert abs(np.mean(hds)-result['per_image_mean']['hd95_pixels'])<1e-9
 assert all(sha(p)==h for p,h in protected.items()),'frozen source or weight drift'
 save(target/'prediction_manifest.json',pngs);save(work/'artifact_audit.json',dict(complete=True,predictions=len(ids),csv_unique_ids=len(by),visualizations=12,independent_PNG_target_confusion_HD95=True,global_metrics_reconciled=metrics,protected_unchanged=True,protected_hashes_after=protected))
 (target/'report.txt').write_text(e['name']+'\n四视图TTA；原无TTA验证best与原测试集合不变；Global指标；HD95单位256网格像素。\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 state('complete',completed_images=len(ds),summary=str(target/'summary.json'),dice_percent=100*result['global']['dice']);pred.close();print('ARM_COMPLETE',ident,100*result['global']['dice'],flush=True)
if __name__=='__main__':
 try:main(sys.argv[1])
 except BaseException as ex:
  save(OUT/sys.argv[1]/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc(),pid=os.getpid()));save(OUT/sys.argv[1]/'status.json',dict(status='blocked',error=repr(ex),pid=os.getpid(),updated_unix=time.time()));raise

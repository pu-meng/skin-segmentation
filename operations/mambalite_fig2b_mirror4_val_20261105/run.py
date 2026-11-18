import os,sys,json,hashlib,time,csv,subprocess,traceback,fcntl,signal,shutil
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
OPS=ROOT/'operations/mambalite_fig2b_mirror4_val_20261105'
OUT=ROOT/'log/mambalite_fig2b_mirror4_val_20261105'
SOURCE=ROOT/'source/mambalite_paper_candidate_20261105'
ORIG=ROOT/'Models/mambalite_fig2b_20261105/seed42/best.pth'
TRAIN=ROOT/'log/mambalite_fig2b_20261105/seed42'
CKPT=OUT/'snapshot/best.pth'
SPLIT_SHA='0f5e2df296f50afb74002ce94438a5b7a17782deaef671f93673a60a8e8b5ed8'
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(SOURCE)]
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from PIL import Image
from candidate_model import build_model,api
from ham_fast_pipeline import PlannedDataset
from project_paths import PATHS

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def identities():
 c=json.loads((SOURCE/'protocol.json').read_text())
 files=set(map(Path,c['source_hashes']))
 files.update([SOURCE/'protocol.json',ROOT/'operations/mambalite_finite_resume_20261105/experiment_revision.py',ROOT/'source/comparison/project_paths.py',Path(PATHS['active_split']),CKPT,OUT/'protocol.json',OUT/'snapshot/selection.json',Path(__file__)])
 for p,h in c['source_hashes'].items():assert sha(p)==h,('Original frozen source mismatch',p)
 return {str(p):sha(p) for p in sorted(files)}
def load():
 m=build_model();s=torch.load(CKPT,map_location='cpu')
 assert s and all(isinstance(v,torch.Tensor) for v in s.values())
 msg=m.load_state_dict(s,strict=True);assert not torch.cuda.is_initialized()
 return m.eval().requires_grad_(False),str(msg)
def prepare():
 assert not OUT.exists(),'Refuse existing output'
 OUT.mkdir();(OUT/'snapshot').mkdir()
 success=False
 for attempt in range(5):
  b=(TRAIN/'selection.json').read_bytes();sel=json.loads(b);h=sha(ORIG)
  shutil.copyfile(ORIG,CKPT)
  a=(TRAIN/'selection.json').read_bytes()
  if b==a and sha(CKPT)==sha(ORIG)==h==sel['checkpoint_sha256']:
   success=True;break
  time.sleep(.5)
 assert success,'Checkpoint/selection race unresolved'
 (OUT/'snapshot/selection.json').write_bytes(b)
 write(OUT/'snapshot/acquisition.json',{'selection':sel,'attempt':attempt+1,'snapshot_sha256':h,'source_sha_before_after_snapshot_equal':True,'acquired_at':time.time(),'training_status_at_acquisition':json.loads((TRAIN/'status.json').read_text())})
 assert sha(PATHS['active_split'])==SPLIT_SHA
 write(OUT/'protocol.json',{'identity':'MambaLiteUNet Figure2(b) reconstruction NOT strict official reproduction','seed':42,'split':'val','n':1010,'checkpoint':'immutable current-best snapshot; intermediate training, not 300 epoch final','selected_epoch':sel['epoch'],'checkpoint_sha256':h,'split_sha256':SPLIT_SHA,'training_entry':str(ROOT/'operations/mambalite_finite_resume_20261105/experiment_revision.py'),'training_protocol':str(SOURCE/'protocol.json'),'tta_reference':str(ROOT/'operations/nnunet_mirror_validation/mirror_core.py'),'views':['identity','horizontal axis3','vertical axis2','both axes2,3'],'fusion':'real final Conv2d hook -> original bilinear interpolation scale_factor2 align_corners=True -> inverse flip -> FP32 logit arithmetic mean -> sigmoid','threshold':.5,'scale_or_postprocessing':False,'batch':1,'workers':0,'eval':True,'no_grad':True,'precision':'FP32','allocator_fraction':.20,'physical_gpu':0,'gpu_uuid':'GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde','shared_authorized':True,'deadline_seconds':7200,'failure_policy':'fail blocked; no precision change, CPU fallback or OOM retry','test_data_accessed':False,'changes_tolerance':1e-6})
 m,msg=load()
 write(OUT/'cpu_preflight.json',{'passed':True,'strict_load':msg,'cuda_initialized':False,'hashes_before':identities(),'no_CPU_forward':True})
 print('CPU_STRICT_LOAD_PASSED',sel,flush=True)
class Predictor(torch.nn.Module):
 def __init__(self,m,tta,mode):
  super().__init__();self.model=m;self.tta=tta;self.mode=mode;self.raw=None;self.n=0;self.times=[];self.max_error=0.;self.verify=4
  self.h=m.final.register_forward_hook(self.capture)
 def capture(self,m,i,o):self.raw=o
 def logits(self,x):
  self.raw=None;prob=self.model(x);assert self.raw is not None
  logit=F.interpolate(self.raw,scale_factor=2,mode='bilinear',align_corners=True)
  if self.verify:
   torch.testing.assert_close(logit.sigmoid(),prob,rtol=0,atol=0)
   self.max_error=max(self.max_error,float((logit.sigmoid()-prob).abs().max()));self.verify-=1
  return logit
 def forward(self,x):
  torch.cuda.synchronize();start=time.perf_counter();p=self.logits(x)
  if self.tta:
   for axes in [(3,),(2,),(2,3)]:p=p+torch.flip(self.logits(torch.flip(x,axes)),axes)
   p=p/4
  p=p.sigmoid();assert bool(torch.isfinite(p).all());torch.cuda.synchronize();self.times.append(time.perf_counter()-start);self.n+=1
  if self.n==1 or self.n%25==0 or self.n==1010:
   write(OUT/'status.json',{'status':'evaluating','mode':self.mode,'forwards_completed':self.n,'expected':1010,'pid':os.getpid(),'updated_at':time.time()});print('FORWARD',self.mode,self.n,'/1010',flush=True)
  return p
 def close(self):self.h.remove()
def timeout(s,f):raise TimeoutError('2h deadline')
def run():
 signal.signal(signal.SIGALRM,timeout);signal.alarm(7200)
 pre=json.loads((OUT/'cpu_preflight.json').read_text());assert pre['passed'] and identities()==pre['hashes_before']
 protocol=json.loads((OUT/'protocol.json').read_text());m,msg=load()
 torch.set_num_threads(4);api.seed_all(42);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.20,0)
 data=api.HAMDataset(Path(PATHS['active_split']).parent,'val');planned=PlannedDataset(Path(PATHS['active_split']).parent,'val')
 assert len(data)==len(planned)==1010 and not data.training and data.rows==planned.rows
 ids=[r['image_id'] for r in data.rows];assert len(set(ids))==1010
 # Exact deterministic training-validation Dataset equivalence for every val input and mask.
 for i in range(1010):
  a,b=data[i],planned[i];assert torch.equal(a[0],b[0]) and torch.equal(a[1],b[1]) and a[2:]==b[2:]
 del planned
 write(OUT/'validation_identity.json',{'ids':ids,'images':1010,'training':False,'all_inputs_targets_exactly_equal_training_PlannedDataset':True,'split_sha256':SPLIT_SHA,'cache_access':'shared mmap contains other splits; only val rows indexed'})
 loader=DataLoader(data,batch_size=1,num_workers=0,shuffle=False)
 write(OUT/'gpu_before.json',{'pid':os.getpid(),'snapshot':subprocess.check_output(['nvidia-smi'],text=True)})
 m.cuda().eval();buffers={n:b.cpu().clone() for n,b in m.named_buffers()}
 pred=Predictor(m,False,'preflight').eval();errors=[]
 with torch.no_grad():
  for i in range(3):
   x=data[i][0].unsqueeze(0).cuda();original=m(x);wrapped=pred(x);torch.testing.assert_close(original,wrapped,rtol=0,atol=0);errors.append(float((original-wrapped).abs().max()))
 pred.close()
 gpuapps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
 assert any(protocol['gpu_uuid'] in l and str(os.getpid())==l.split(',')[1].strip() for l in gpuapps.splitlines()),gpuapps
 write(OUT/'gpu_preflight.json',{'passed':True,'pid':os.getpid(),'gpu_uuid':protocol['gpu_uuid'],'compute_apps':gpuapps,'first_real_val_ids':ids[:3],'noTTA_original_probability_exact_errors':errors,'logit_hook':'model.final -> F.interpolate scale2 bilinear align_corners=True; no inverse sigmoid','FP32':all(p.dtype==torch.float32 for p in m.parameters())})
 print('GPU_PREFLIGHT_PASSED',os.getpid(),protocol['gpu_uuid'],errors,flush=True)
 results={};audits={};allrows={}
 for mode in ['none','mirror4']:
  path=OUT/mode;assert not path.exists();pred=Predictor(m,mode=='mirror4',mode).eval();torch.cuda.reset_peak_memory_stats()
  result=api.evaluate(pred,loader,api.BceDiceLoss().cuda(),torch.device('cuda'),path);assert pred.n==1010
  rows=list(csv.DictReader((path/'per_image_metrics.csv').open()));byid={r['image_id']:r for r in rows}
  assert len(rows)==len(byid)==1010 and set(byid)==set(ids)
  assert {p.name for p in (path/'predictions').glob('*.png')}=={i+'_segmentation.png' for i in ids}
  assert len(list((path/'visualizations').glob('*.png')))==12
  totals=np.zeros(4,dtype=np.int64);manifest={}
  for row in data.rows:
   ident=row['image_id'];png=path/'predictions'/(ident+'_segmentation.png');p=np.asarray(Image.open(png))>0;y=data.masks[int(row['cache_index'])]>0
   counts=api.confusion(p,y);totals+=counts
   assert all(int(byid[ident][k])==int(v) for k,v in zip(['tp','fp','fn','tn'],counts))
   assert abs(api.hd95(p,y)-float(byid[ident]['hd95_pixels']))<1e-9
   manifest[png.name]=sha(png)
  for k,v in zip(['tp','fp','fn','tn'],totals):assert int(result['global'][k])==int(v)==sum(int(r[k]) for r in rows)
  for k in ['dice','iou','hd95_pixels']:assert abs(np.mean([float(r[k]) for r in rows])-result['per_image_mean'][k])<1e-12
  assert all(torch.equal(buffers[n],b.cpu()) for n,b in m.named_buffers())
  assert identities()==pre['hashes_before']
  result.update(mode=mode,split='val',seed=42,selected_epoch=protocol['selected_epoch'],checkpoint_sha256=sha(CKPT),split_sha256=SPLIT_SHA,intermediate_training_snapshot=True,test_data_accessed=False,logit_probability_max_error=pred.max_error,forward_seconds=sum(pred.times),timing_caveat='shared GPU, sequential modes, not isolated benchmark',peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
  write(path/'summary.json',result);write(path/'prediction_manifest.json',manifest)
  results[mode]=result;allrows[mode]=byid;audits[mode]={'predictions':1010,'unique_csv_ids':1010,'visualizations':12,'independent_PNG_GT_confusion_HD95_verified':True,'csv_sha256':sha(path/'per_image_metrics.csv')}
  write(OUT/'partial_results.json',results);print('COMPLETE',mode,json.dumps(result),flush=True);pred.close()
 changes=[]
 for ident in ids:
  a=float(allrows['none'][ident]['dice']);b=float(allrows['mirror4'][ident]['dice']);d=b-a
  changes.append({'image_id':ident,'none_dice':a,'mirror4_dice':b,'dice_delta':d,'classification':'corrected' if d>1e-6 else 'damaged' if d< -1e-6 else 'unchanged'})
 with (OUT/'image_dice_changes.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(changes[0]));w.writeheader();w.writerows(changes)
 comparison={'global_dice_delta_percentage_points':100*(results['mirror4']['global']['dice']-results['none']['global']['dice']),'global_iou_delta_percentage_points':100*(results['mirror4']['global']['iou']-results['none']['global']['iou']),'mean_hd95_delta_pixels':results['mirror4']['per_image_mean']['hd95_pixels']-results['none']['per_image_mean']['hd95_pixels'],'image_counts':{k:sum(r['classification']==k for r in changes) for k in ['corrected','damaged','unchanged']},'dice_tolerance':1e-6}
 after=identities();assert after==pre['hashes_before']
 write(OUT/'audit.json',{'complete':True,'groups':audits,'hashes_before':pre['hashes_before'],'hashes_after':after,'sources_snapshot_split_unchanged':True,'BN_buffers_unchanged':True,'pid':os.getpid(),'test_data_accessed':False})
 write(OUT/'results.json',{'complete':True,'results':results,'comparison':comparison,'protocol':protocol})
 lines=['# MambaLiteUNet 图2(b)重建版 seed42：验证集配对TTA评估','NOT strict official reproduction；固定中途best快照，非300轮最终结果。仅val1010图像，不是患者；不访问test，不训练或重选权重。',f"选中epoch {protocol['selected_epoch']}；快照SHA256 `{sha(CKPT)}`。",'四视图：原图/水平/垂直/双翻转，真实logits逆翻平均后sigmoid，阈值0.5；FP32 batch1，无尺度或后处理。','','|模式|Global Dice %|Global IoU %|mean HD95 像素|','|---|---:|---:|---:|']
 for mode in ['none','mirror4']:
  r=results[mode];lines.append(f"|{mode}|{100*r['global']['dice']:.6f}|{100*r['global']['iou']:.6f}|{r['per_image_mean']['hd95_pixels']:.6f}|")
 lines+=['',json.dumps(comparison,ensure_ascii=False),'纠正/破坏/不变为逐图Dice差超过±1e-6；非像素纠正数。','两组各1010预测PNG、逐图CSV、12可视化；全部预测与真实val masks独立复算混淆计数及HD95通过。','results.json / audit.json / snapshot/acquisition.json；共享GPU时序计时不作公平加速比。']
 (OUT/'report.md').write_text('\n'.join(lines)+'\n')
 write(OUT/'status.json',{'status':'complete','pid':os.getpid(),'groups':2,'images_per_group':1010,'test_data_accessed':False});print('ALL_COMPLETE',comparison,flush=True)
if __name__=='__main__':
 try:
  if sys.argv[1]=='prepare':prepare()
  else:
   with (OUT/'RUNNING.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (OUT/'run_started.json').exists(),'Refuse repeat'
    write(OUT/'run_started.json',{'pid':os.getpid(),'time':time.time()});(OUT/'pid').write_text(str(os.getpid())+'\n');run()
 except BaseException as e:
  if OUT.exists():write(OUT/'failure.json',{'pid':os.getpid(),'error':repr(e),'traceback':traceback.format_exc()});write(OUT/'status.json',{'status':'failed_blocked','pid':os.getpid(),'error':repr(e)})
  raise

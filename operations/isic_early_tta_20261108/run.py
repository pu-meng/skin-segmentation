"""Fixed-best mirror4 inference for the 12 completed ISIC2018 comparisons."""
import os,sys,json,csv,hashlib,time,fcntl,subprocess,traceback
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
OPS=Path(__file__).resolve().parent
LOG=ROOT/'log/ISIC2018_瓶颈对比_20261031'
OUT=LOG/'mirror4_TTA_20261108'
PY='/home/PuMengYu/anaconda3/envs/medseg/bin/python'
UUIDS=['GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde','GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b']
ARMS=['tinyvim_base','tinyvim_v1','tinyvim_v2','tinyvim_v2_ds_proj','tinyvim_v2_no_proj','tinyvim_v3','unet_base','unet_v1','unet_v2','unet_v2_ds_proj','unet_v2_no_proj','unet_v3']
ASSIGN={a:i%2 for i,a in enumerate(ARMS)}
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba'),str(ROOT/'source/isic2018')]
from source.isic2018 import data
from source.isic2018 import experiment as builder

def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def protected_sources():
 protocol=read(LOG/'protocol.json')
 files={str((ROOT/rel).resolve()):digest for rel,digest in protocol['source_sha256'].items() if rel!='source/bottleneck_mamba/mamba_figure_layout.py'}
 for p in [Path(__file__),ROOT/'source/isic2018/data.py',ROOT/'source/isic2018/experiment.py']:
  files[str(p.resolve())]=sha(p)
 return files

def verify_sources(protocol):
 for f,h in protocol['protected_sha256'].items():assert sha(f)==h,'source drift '+f

def prepare():
 assert not OUT.exists(),'Refuse to overwrite an existing TTA result root'
 data.verify();protocol=read(LOG/'protocol.json');assert protocol['dataset']=='ISIC2018'
 assert protocol['data']['counts']==dict(train=1780,val=258,test=518)
 records={}
 for arm in ARMS:
  run=LOG/'runs'/arm;status=read(run/'test_status.json');audit=read(run/'test_artifact_audit.json')
  summary=read(run/'test/summary.json');checkpoint=Path(status['checkpoint'])
  assert status['status']=='test_complete' and status['completed_images']==518
  assert audit['complete'] and audit['prediction_count']==518
  assert status['history_rows']==300 and summary['split']=='test' and summary['images']==518
  assert summary['selected_epoch']==status['selected_epoch']
  assert sha(checkpoint)==status['checkpoint_sha256']==summary['checkpoint_sha256']
  pred=run/'test/predictions';assert len(list(pred.glob('*_segmentation.png')))==518
  records[arm]=dict(checkpoint=str(checkpoint),checkpoint_sha256=status['checkpoint_sha256'],selected_epoch=status['selected_epoch'],old_no_tta_summary=str(run/'test/summary.json'),old_no_tta_predictions=str(pred),no_tta_artifact_audit=str(run/'test_artifact_audit.json'),dataset='ISIC2018',split='test',test_images=518)
 OUT.mkdir(parents=True)
 manifest=dict(identity='ISIC2018 12 fixed-best four-view TTA supplements; no retraining',dataset='ISIC2018',split='test',split_sha256=protocol['data']['split_sha256'],dedup_split_sha256=protocol['data']['dedup_split_sha256'],counts=protocol['data']['counts'],models=records,protected_sha256=protected_sources(),protocol=dict(views=['identity','horizontal flip','vertical flip','both flips'],fusion='inverse-transform each view; arithmetic mean of pre-sigmoid logits; sigmoid once; threshold >=0.5',selection='reuse exact existing minimum-validation-loss checkpoint; test labels do not select weights',no_tta_audit='for all 518 cases, re-evaluated identity-view masks must exactly equal the archived no-TTA prediction masks',batch_size=1,precision='FP32',retraining=False),scheduling=dict(user_requested_shared_execution=True,at_most_one_added_inference_per_gpu=True,free_memory_gate_mib=4096,max_compute_pids_per_gpu=5,admission_lock=str(ROOT/'operations/encoder_swap_3slot_takeover_20261108/admission.lock')))
 write(OUT/'protocol.json',manifest);write(OUT/'prepare_complete.json',dict(passed=True,models=ARMS,count=12,protocol_sha256=sha(OUT/'protocol.json')))
 print('PREPARED_12_FIXED_CHECKPOINTS',flush=True)

def verify():
 p=read(OUT/'protocol.json');verify_sources(p);data.verify()
 for arm,r in p['models'].items():assert sha(r['checkpoint'])==r['checkpoint_sha256'],'checkpoint drift '+arm
 return p

class MirrorPredictor:
 def __init__(self,torch,model,head,ids,old_predictions):
  from torch import nn
  from torch.nn import functional as F
  import numpy as np
  from PIL import Image
  class Wrapped(nn.Module):
   def __init__(self):
    super().__init__();self.model=model;self.raw=None;self.index=0;self.ids=ids;self.old_predictions=old_predictions;self.verify_remaining=4;self.baseline_matches=0
    self.handle=head.register_forward_hook(self.capture)
   def capture(self,module,inputs,output):self.raw=output
   def raw_logits(self,x):
    self.raw=None;prob=self.model(x);assert self.raw is not None
    logit=self.raw
    assert logit.shape==prob.shape==(x.shape[0],1,256,256),(logit.shape,prob.shape)
    if self.verify_remaining:
     torch.testing.assert_close(torch.sigmoid(logit),prob,rtol=0,atol=0);self.verify_remaining-=1
    return logit,prob
   def forward(self,x):
    logits,base_prob=self.raw_logits(x)
    assert x.shape[0]==1
    from PIL import Image
    expected=np.asarray(Image.open(self.old_predictions/(self.ids[self.index]+'_segmentation.png')).convert('L'))>0
    current=base_prob[0,0].detach().cpu().numpy()>=0.5
    assert np.array_equal(current,expected),'No-TTA prediction changed: '+self.ids[self.index]
    self.baseline_matches+=1;self.index+=1
    for axes in [(2,),(3,),(2,3)]:
     flip_logits,_=self.raw_logits(torch.flip(x,axes));logits=logits+torch.flip(flip_logits,axes)
    return torch.sigmoid(logits/4)
   def close(self):self.handle.remove()
  self.wrapped=Wrapped()

def phase(arm,gpu):
 import torch,numpy as np
 from torch import nn
 from torch.utils.data import DataLoader
 from PIL import Image
 import ham_fast_runner as runner
 p=verify();assert p['models'][arm]['split']=='test'
 work=OUT/arm;target=work/'mirror4';work.mkdir(parents=True,exist_ok=True);assert not target.exists()
 api=runner.baseline_api();api.seed_all(42);torch.set_num_threads(2)
 model=builder.build(arm,pretrained=False);heads=[m for m in model.modules() if isinstance(m,nn.Conv2d) and m.out_channels==1 and m.kernel_size==(1,1)]
 assert len(heads)==1,[(type(m).__name__,m.out_channels) for m in heads]
 state=torch.load(p['models'][arm]['checkpoint'],map_location='cpu',weights_only=False);load=model.load_state_dict(state,strict=True);assert not torch.cuda.is_initialized()
 write(work/'cpu_load_audit.json',dict(strict_load=True,checkpoint=p['models'][arm]['checkpoint'],checkpoint_sha256=sha(p['models'][arm]['checkpoint']),missing_keys=list(load.missing_keys),unexpected_keys=list(load.unexpected_keys),cuda_initialized=False,selected_epoch=p['models'][arm]['selected_epoch']))
 torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.08,0);device=torch.device('cuda:0');model.to(device).eval()
 ds=data.ISICDataset(data.CACHE,'test');assert len(ds)==518 and not ds.training
 ids=[r['image_id'] for r in ds.rows];assert len(set(ids))==518
 loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=0)
 wrapper=MirrorPredictor(torch,model,heads[0],ids,Path(p['models'][arm]['old_no_tta_predictions'])).wrapped.eval()
 info=dict(status='evaluating_mirror4',arm=arm,gpu=gpu,pid=os.getpid(),checkpoint_sha256=p['models'][arm]['checkpoint_sha256'],selected_epoch=p['models'][arm]['selected_epoch'],test_images=518,no_tta_mask_comparison='pending',started_unix=time.time())
 write(work/'status.json',info);criterion=api.BceDiceLoss().to(device)
 try:
  result=api.evaluate(wrapper,loader,criterion,device,target)
  assert wrapper.index==wrapper.baseline_matches==518 and wrapper.verify_remaining==0
  assert len(list((target/'predictions').glob('*_segmentation.png')))==518
  with (target/'per_image_metrics.csv').open(newline='') as f:rows=list(csv.DictReader(f))
  assert len(rows)==518 and {r['image_id'] for r in rows}==set(ids)
  for key in ['tp','fp','fn','tn']:assert sum(int(r[key]) for r in rows)==result['global'][key]
  result.update(dataset='ISIC2018',model=arm,split='test',tta='mirror4',seed=42,selected_epoch=p['models'][arm]['selected_epoch'],checkpoint=p['models'][arm]['checkpoint'],checkpoint_sha256=p['models'][arm]['checkpoint_sha256'],split_sha256=p['split_sha256'],test_used_for_checkpoint_selection=False,no_tta_reference_exact_masks=wrapper.baseline_matches,tta_protocol=p['protocol']['fusion'])
  write(target/'summary.json',result)
  audit=dict(complete=True,dataset='ISIC2018',split='test',prediction_count=518,metric_count=518,baseline_no_tta_exact_mask_matches=wrapper.baseline_matches,logits_probability_exact_checks=4,confusion_reconciled=True,checkpoint_sha256=p['models'][arm]['checkpoint_sha256'],selected_epoch=p['models'][arm]['selected_epoch'],test_used_for_model_selection=False)
  write(target/'artifact_audit.json',audit)
  (target/'report.txt').write_text(f"ISIC2018 {arm} fixed-best mirror4 TTA; selected epoch {p['models'][arm]['selected_epoch']}; test 518 images; Global Dice {100*result['global']['dice']:.4f}%; all 518 identity-view masks exactly match archived no-TTA predictions.\n")
  verify();write(work/'status.json',dict(status='complete',arm=arm,gpu=gpu,pid=os.getpid(),completed_images=518,baseline_no_tta_exact_mask_matches=wrapper.baseline_matches,global_dice_percent=100*result['global']['dice'],checkpoint_sha256=p['models'][arm]['checkpoint_sha256'],finished_unix=time.time()))
  print('COMPLETE',arm,'TTA_GLOBAL_DICE',100*result['global']['dice'],flush=True)
 finally:wrapper.close()

def resources(gpu):
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 stats=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader'],text=True)
 pids=[int(s.split(',')[1]) for s in apps.splitlines() if s.startswith(UUIDS[gpu])]
 free=int(next(s.split(',')[1].strip().split()[0] for s in stats.splitlines() if s.startswith(UUIDS[gpu])))
 return pids,free

def launch(arm,gpu):
 guard=Path(read(OUT/'protocol.json')['scheduling']['admission_lock'])
 while True:
  lock=guard.open('a');fcntl.flock(lock,fcntl.LOCK_EX);pids,free=resources(gpu)
  if len(pids)<5 and free>=4096:break
  fcntl.flock(lock,fcntl.LOCK_UN);lock.close();write(OUT/arm/'waiting.json',dict(status='waiting_for_admission',gpu=gpu,compute_pids=pids,free_mib=free,updated_unix=time.time()));time.sleep(30)
 (OUT/arm).mkdir(parents=True,exist_ok=True)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=UUIDS[gpu],EXPERIMENT_PHYSICAL_GPU=str(gpu),EXPERIMENT_GPU_UUID=UUIDS[gpu],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 log=(OUT/arm/'phase.log').open('x');child=subprocess.Popen([PY,'-B','-u',__file__,'phase',arm,str(gpu)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
 write(OUT/arm/'controller_status.json',dict(status='evaluating',pid=child.pid,gpu=gpu,started_unix=time.time()))
 deadline=time.time()+180
 while child.poll() is None:
  pids,_=resources(gpu)
  if child.pid in pids:break
  if time.time()>deadline:child.terminate();raise TimeoutError(arm+' phase not GPU resident')
  time.sleep(1)
 fcntl.flock(lock,fcntl.LOCK_UN);lock.close();rc=child.wait();log.close();write(OUT/arm/'exitcode.json',dict(exitcode=rc,pid=child.pid,finished_unix=time.time()));assert rc==0,arm+' TTA phase failed'

def worker(gpu):
 p=verify()
 for arm in [a for a in ARMS if ASSIGN[a]==gpu]:
  launch(arm,gpu);assert read(OUT/arm/'status.json')['status']=='complete'
 write(OUT/f'worker_gpu{gpu}_complete.json',dict(complete=True,gpu=gpu,arms=[a for a in ARMS if ASSIGN[a]==gpu]))
 if all((OUT/a/'status.json').exists() and read(OUT/a/'status.json').get('status')=='complete' for a in ARMS):
  write(OUT/'complete.json',dict(complete=True,models=ARMS,split_sha256=p['split_sha256'],finished_unix=time.time()))

if __name__=='__main__':
 try:
  if sys.argv[1]=='prepare':prepare()
  elif sys.argv[1]=='worker':worker(int(sys.argv[2]))
  elif sys.argv[1]=='phase':phase(sys.argv[2],int(sys.argv[3]))
 except BaseException as e:
  if OUT.exists():write(OUT/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),pid=os.getpid(),time=time.time()))
  raise

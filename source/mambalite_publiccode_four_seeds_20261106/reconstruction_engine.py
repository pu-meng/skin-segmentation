"""Isolated authorized five-seed reconstruction experiment; no legacy writes."""
import os,sys,csv,json,hashlib,random,time,subprocess,argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader,Sampler
sys.path.insert(0,'/home/PuMengYu/experiments/skin_segmentation/source/mambalite_paper_candidate_20261105')
from candidate_model import build_model,SS2D,api,HERE,ROOT
from ham_fast_pipeline import PlannedDataset,worker_init
from project_paths import CACHE
from PIL import Image
LOG=ROOT/'log/mambalite_publiccode_four_seeds_20261106/fig2b'
MODELS=ROOT/'Models/mambalite_publiccode_four_seeds_20261106/fig2b'
SPLIT=ROOT/'data/processed/ham10000_group42_dedup_20261023/split.csv'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');temp.replace(p)
def config():return json.loads((HERE/'protocol.json').read_text())
def verify_sources():
 c=config()
 for p,h in c['source_hashes'].items():
  if sha(p)!=h:raise RuntimeError('Frozen source changed: '+p)
 if sha(SPLIT)!=c['data']['split_sha256']:raise RuntimeError('Frozen split changed')
 rows=list(csv.DictReader(SPLIT.open()))
 assert {s:sum(r['split']==s for r in rows) for s in ['train','val','test']}==c['data']['counts']
 for name,h in c['cache_hashes'].items():
  if sha(CACHE/name)!=h:raise RuntimeError('Cache changed: '+name)
 return c

def gpu_snapshot():
 return subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True)

class SeedPlanSampler(Sampler):
 """Private shuffle + augmentation streams, never the shared seed42 loader."""
 def __init__(self,dataset,seed):
  self.n=len(dataset);self.generator=torch.Generator().manual_seed(seed)
  self.py=random.Random(seed);self.np=np.random.RandomState(seed)
 def __len__(self):return self.n
 def __iter__(self):
  for index in torch.randperm(self.n,generator=self.generator).tolist():
   flip=self.py.random()>.5;k=int(self.np.randint(0,4)) if flip else 0
   axis=int(self.np.randint(0,2)) if flip else 0
   rotate=self.py.random()>.5;angle=int(self.np.randint(20,80)) if rotate else 0
   yield index,flip,k,axis,rotate,angle
 def state(self):return dict(torch=self.generator.get_state(),python=self.py.getstate(),numpy=self.np.get_state())
def sampler_signature(seed):
 class Small:
  def __len__(self):return 24
 return list(SeedPlanSampler(Small(),seed))
def make_loaders(seed,workers=2,include_test=False):
 loaders={}
 for split in (['test'] if include_test else ['train','val']):
  ds=PlannedDataset(SPLIT.parent,split)
  kwargs=dict(batch_size=8 if split=='train' else 1,
   sampler=SeedPlanSampler(ds,seed) if split=='train' else None,num_workers=workers,
   pin_memory=True,drop_last=False,worker_init_fn=worker_init,
   generator=torch.Generator().manual_seed(seed+10000+(split!='train')))
  if workers:kwargs.update(persistent_workers=True,prefetch_factor=2,multiprocessing_context='spawn')
  loaders[split]=DataLoader(ds,**kwargs)
 return loaders

def state_digest(m):
 h=hashlib.sha256()
 for n,p in m.state_dict().items():h.update(n.encode());h.update(p.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def model_path(seed):return MODELS/f'seed{seed}'
def run_path(seed):return LOG/f'seed{seed}'
def strict_gate(seed):
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 verify_sources();p=model_path(seed)/'best.pth';m=build_model()
 m.load_state_dict(torch.load(p,map_location='cpu'),strict=True)
 assert not torch.cuda.is_initialized()
 save(run_path(seed)/'cpu_strict_load.json',{'status':'passed','seed':seed,'checkpoint':str(p),'sha256':sha(p),'strict':True,'CUDA_initialized':False,'parameters':sum(p.numel() for p in m.parameters())})
 print('CPU STRICT BEST LOAD PASSED seed',seed,flush=True)

def train(seed):
 c=verify_sources()
 for name in ['cpu','gpu']:
  a=json.loads((HERE/(name+'_acceptance.json')).read_text())
  assert a['status']=='passed' and a['source_hashes']==c['source_hashes']
 assert os.environ['CUDA_VISIBLE_DEVICES']=='1'
 rd=run_path(seed);md=model_path(seed)
 resume=(seed==42)
 rd.mkdir(parents=True,exist_ok=True);md.mkdir(parents=True,exist_ok=resume)
 if not resume:assert not (rd/'history.csv').exists(),'fresh training refuses old history'
 api.seed_all(seed);torch.set_num_threads(4)
 device=torch.device('cuda:0');m=build_model()
 opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8)
 scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=50,eta_min=1e-5)
 criterion=api.BceDiceLoss().to(device);loaders=make_loaders(seed)
 updates=0;best=float('inf');best_epoch=0;initial=state_digest(m)
 provenance={'seed':seed,'initial_state_sha256':initial,'initialization':'independent fresh random model, preflight weights discarded','pid':os.getpid(),'physical_gpu':1,'protocol_sha256':sha(HERE/'protocol.json'),'sources':c['source_hashes'],'split_sha256':sha(SPLIT),'parameters':sum(p.numel() for p in m.parameters()),'training':c['training'],'gpu_start':gpu_snapshot()}
 start_epoch=1
 if resume:
  from reconstruction_restore import restore_cpu
  payload,audit=restore_cpu(m,opt,scheduler,loaders,seed,rd,md)
  updates=audit['optimizer_updates'];best=payload['best_loss'];best_epoch=payload['best_epoch'];initial=payload['initial_state_sha256'];start_epoch=payload['epoch']+1
 else:save(rd/'provenance.json',provenance)
 m.to(device)
 for state in opt.state.values():
  for key,value in state.items():
   if torch.is_tensor(value) and key!='step':state[key]=value.to(device)
 if resume:
  random.setstate(payload['python_rng']);np.random.set_state(payload['numpy_rng']);torch.set_rng_state(payload['torch_rng']);torch.cuda.set_rng_state(payload['cuda_rng'])
  save(rd/'resume_state_audit.json',audit)
  print('RESUME STATE RESTORED',json.dumps(audit),flush=True)
 else:print('FRESH TRAIN START',json.dumps(provenance),flush=True)
 for epoch in range(start_epoch,301):
  m.train();t=time.monotonic();losses=[];lr=opt.param_groups[0]['lr']
  for step,(x,y,ids,_) in enumerate(loaders['train'],1):
   opt.zero_grad(set_to_none=True);p=m(x.to(device,non_blocking=True))
   loss=criterion(p,y.to(device,non_blocking=True))
   if not bool(torch.isfinite(loss)):raise RuntimeError(f'Nonfinite training loss epoch{epoch} step{step}')
   loss.backward()
   checks=[torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None]
   if checks and not bool(torch.stack(checks).all()):raise RuntimeError('Nonfinite training gradient')
   opt.step();updates+=1;losses.append(float(loss.detach()))
   if step==1 or step%50==0 or step==len(loaders['train']):
    s={'status':'training','seed':seed,'pid':os.getpid(),'epoch':epoch,'epoch_completed':epoch-1,'step':step,'steps_per_epoch':len(loaders['train']),'optimizer_updates':updates,'train_loss':float(np.mean(losses)),'updated_at':datetime.now().isoformat()}
    save(rd/'status.json',s)
    print('TRAIN',json.dumps(s),flush=True)
  # Exact per-image validation with the inherited evaluator, not a fast batch approximation.
  validation=api.evaluate(m,loaders['val'],criterion,device)
  assert validation['images']==1010 and np.isfinite(validation['loss'])
  if validation['loss']<best:
   best=validation['loss'];best_epoch=epoch
   temp=md/'best.tmp.pth';torch.save(m.state_dict(),temp);temp.replace(md/'best.pth')
   save(rd/'selection.json',{'rule':'minimum complete validation segmentation loss, earliest tie','epoch':epoch,'val_loss':best,'checkpoint':str(md/'best.pth'),'checkpoint_sha256':sha(md/'best.pth')})
  scheduler.step()
  payload=dict(epoch=epoch,model=m.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),best_loss=best,best_epoch=best_epoch,seed=seed,initial_state_sha256=initial,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),sampler_rng=loaders['train'].sampler.state(),protocol_sha256=sha(HERE/'protocol.json'),split_sha256=sha(SPLIT))
  tmp=md/'latest.tmp.pth';torch.save(payload,tmp);tmp.replace(md/'latest.pth')
  row=dict(epoch=epoch,train_loss=float(np.mean(losses)),val_loss=validation['loss'],val_global_dice=validation['global']['dice'],val_global_iou=validation['global']['iou'],lr=lr,seconds=time.monotonic()-t,best_epoch=best_epoch,optimizer_updates=updates)
  with (rd/'history.csv').open('a',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(row))
   if epoch==1:w.writeheader()
   w.writerow(row)
  s.update(epoch_completed=epoch,validation=validation,best_epoch=best_epoch,updated_at=datetime.now().isoformat());save(rd/'status.json',s)
  print('EPOCH COMPLETE',json.dumps(row),flush=True)
 assert updates==300*len(loaders['train'])
 torch.save(m.state_dict(),md/'final.pth')
 history=list(csv.DictReader((rd/'history.csv').open()))
 assert [int(r['epoch']) for r in history]==list(range(1,301))
 s.update(status='training_complete_test_pending',epoch_completed=300,optimizer_updates=updates)
 save(rd/'status.json',s);save(rd/'training_complete.json',{'complete':True,'seed':seed,'epochs':300,'updates':updates,'best_epoch':best_epoch,'best_sha256':sha(md/'best.pth')})
 print('TRAINING COMPLETE seed',seed,flush=True)

def test(seed):
 c=verify_sources();rd=run_path(seed);md=model_path(seed)
 tr=json.loads((rd/'training_complete.json').read_text());assert tr['complete'] and tr['epochs']==300
 gate=json.loads((rd/'cpu_strict_load.json').read_text());assert gate['status']=='passed' and gate['sha256']==sha(md/'best.pth')==tr['best_sha256']
 api.seed_all(seed);torch.set_num_threads(4);device=torch.device('cuda:0')
 m=build_model().to(device);m.load_state_dict(torch.load(md/'best.pth',map_location=device),strict=True)
 loader=make_loaders(seed,include_test=True)['test'];out=rd/'test'
 assert not out.exists(),'no test overwrite'
 save(rd/'status.json',{'status':'testing_best','seed':seed,'pid':os.getpid(),'test_images':2020})
 print('BEST TEST START seed',seed,'2020 images',flush=True)
 summary=api.evaluate(m,loader,api.BceDiceLoss().to(device),device,out)
 expected={r['image_id'] for r in loader.dataset.rows}
 records=list(csv.DictReader((out/'per_image_metrics.csv').open()))
 assert len(expected)==len(records)==summary['images']==2020
 assert {r['image_id'] for r in records}==expected
 assert {p.name for p in (out/'predictions').glob('*.png')}=={i+'_segmentation.png' for i in expected}
 assert len(list((out/'visualizations').glob('*.png')))==12
 # Independent readback of every PNG and cached target, not merely CSV arithmetic.
 by_id={r['image_id']:r for r in records};total=np.zeros(4,dtype=np.int64)
 for row in loader.dataset.rows:
  p=np.asarray(Image.open(out/'predictions'/(row['image_id']+'_segmentation.png')))>0
  y=loader.dataset.masks[int(row['cache_index'])]>0
  count=api.confusion(p,y);total+=count
  assert all(int(by_id[row['image_id']][k])==int(v) for k,v in zip(['tp','fp','fn','tn'],count))
  hd=api.hd95(p,y);assert abs(hd-float(by_id[row['image_id']]['hd95_pixels']))<1e-9
 assert all(int(summary['global'][k])==int(v) for k,v in zip(['tp','fp','fn','tn'],total))
 summary.update(identity=c['identity'],seed=seed,dataset='HAM10000',split='test',split_file=str(SPLIT),split_sha256=sha(SPLIT),checkpoint=str(md/'best.pth'),checkpoint_sha256=sha(md/'best.pth'),selected_epoch=tr['best_epoch'],protocol_sha256=sha(HERE/'protocol.json'),source_hashes=c['source_hashes'])
 save(out/'summary.json',summary)
 (out/'report.txt').write_text('MambaLiteUNet Figure2(b) reconstruction, NOT strict author reproduction\nHAM10000 test n=2020; no TTA; threshold0.5; fixed validation-best\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
 audit={'complete':True,'seed':seed,'predictions':2020,'csv_unique_ids':2020,'visualizations':12,'independent_prediction_target_count_and_HD95_audit':True,'checkpoint_sha256':sha(md/'best.pth'),'split_sha256':sha(SPLIT)}
 save(rd/'artifact_audit.json',audit);save(rd/'status.json',dict(audit,status='complete',pid=os.getpid()))
 print('BEST TEST COMPLETE',json.dumps(audit),flush=True)

def aggregate():
 records=[];initials=[]
 for seed in range(42,47):
  rd=run_path(seed);a=json.loads((rd/'artifact_audit.json').read_text());assert a['complete'] and a['seed']==seed
  r=json.loads((rd/'test/summary.json').read_text());assert r['images']==2020 and r['seed']==seed
  assert r['checkpoint_sha256']==sha(model_path(seed)/'best.pth')
  initials.append(json.loads((rd/'provenance.json').read_text())['initial_state_sha256']);records.append(r)
 assert len(set(initials))==5
 result={'identity':config()['identity'],'n_independent_runs':5,'seeds':list(range(42,47)),'std_ddof':1,'global':{},'per_image_mean':{},'sources':[str(run_path(s)/'test/summary.json') for s in range(42,47)]}
 for section,keys in [('global',['dice','iou','accuracy','sensitivity','specificity']),('per_image_mean',['dice','iou','hd95_pixels'])]:
  for k in keys:
   v=np.array([r[section][k] for r in records]);result[section][k]={'mean':float(v.mean()),'sample_std':float(v.std(ddof=1)),'values':v.tolist()}
 save(LOG/'five_run_summary.json',result)
 (LOG/'five_run_report.txt').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print('ALL FIVE INDEPENDENT RUNS COMPLETE',json.dumps(result),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','strict','test','aggregate']);p.add_argument('--seed',type=int,choices=range(42,47));a=p.parse_args()
 try:
  if a.mode=='aggregate':aggregate()
  else:globals()[{'strict':'strict_gate'}.get(a.mode,a.mode)](a.seed)
 except BaseException as e:
  import traceback
  save((run_path(a.seed) if a.seed else LOG)/('failure_'+a.mode+'.json'),{'status':'failed','mode':a.mode,'seed':a.seed,'pid':os.getpid(),'error':repr(e),'traceback':traceback.format_exc()})
  raise

"""Only new isolated outputs; public43-46 and reconstructed42 only."""
import os,sys,csv,json,hashlib,time,io,shutil,statistics
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation');NAME='mambalite_publiccode_four_seeds_20261106'
SRC=ROOT/'source'/NAME;OPS=ROOT/'operations'/NAME;LOG=ROOT/'log'/NAME;MODELS=ROOT/'Models'/NAME
sys.path[:0]=[str(SRC),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t')
import torch,numpy as np
import ham_fast_runner as old
from project_paths import current_protocol,CONFIGS,CACHE
from seed_loaders import make_fast_loaders
from torch.utils.data import DataLoader
from PIL import Image
import final_test_pipeline as ft
api=old.baseline_api()
def sha(p):return ft.sha256_file(p)
def save(p,v):return ft.write_json(p,v)
def config():return json.loads((OPS/'protocol.json').read_text())
def verify():
 c=config()
 for p,h in c['sources'].items():assert sha(p)==h,'source changed '+p
 assert sha(c['split_file'])==c['split_sha256'] and sha(c['test_ids_file'])==c['test_ids_sha256']
 assert len(json.loads(Path(c['test_ids_file']).read_text()))==2020
 assert shutil.disk_usage(ROOT).free>=8*2**30,'disk free <8GiB; blocked, no cleanup'
 return c
def identity(arm):
 if arm.startswith('public'):
  seed=int(arm[6:]);assert seed in (43,44,45,46)
  return seed,LOG/arm,MODELS/arm,False
 assert arm=='fig2b42','only 5 explicit arms authorized'
 return 42,LOG/'fig2b/seed42',MODELS/'fig2b/seed42',True
def build(recon=False):
 if recon:
  from candidate_model import build_model
  return build_model()
 return api.MambaLiteUNet()
def seed_loaders(seed,workers=2):
 c=current_protocol(CONFIGS/'protocol_v5_dedup_seed42.json')
 return make_fast_loaders(c,api,workers=workers,validation_batch=8,seed=seed)
def independent_audit(out,expected,summary):
 audit=ft.validate_outputs(out,expected,summary)
 records=list(csv.DictReader((out/'per_image_metrics.csv').open()));by={r['image_id']:r for r in records}
 rows=[r for r in csv.DictReader(Path(config()['split_file']).open()) if r['split']=='test']
 masks=np.load(CACHE/'masks.npy',mmap_mode='r');tot=np.zeros(4,dtype=np.int64);hds=[]
 for r in rows:
  p=np.asarray(Image.open(out/'predictions'/(r['image_id']+'_segmentation.png')))>0;y=masks[int(r['cache_index'])]>0
  cnt=api.confusion(p,y);tot+=cnt;hd=api.hd95(p,y);hds.append(hd)
  assert all(int(by[r['image_id']][k])==int(v) for k,v in zip(['tp','fp','fn','tn'],cnt))
  assert abs(hd-float(by[r['image_id']]['hd95_pixels']))<1e-9
 assert all(int(summary['global'][k])==int(v) for k,v in zip(['tp','fp','fn','tn'],tot))
 assert abs(float(np.mean(hds))-summary['per_image_mean']['hd95_pixels'])<1e-9
 audit['independent_PNG_target_confusion_HD95']=True
 return audit
def cpu():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 c=verify();torch.set_num_threads(1)
 # No new training is triggered if old seed42 is not reusable.
 rd=Path(c['legacy_seed42']['training_run']);td=Path(c['legacy_seed42']['test_root']);proof={}
 try:
  assert ft.training_ready(rd)
  fixed=ft.fixed_best_record(rd);assert fixed['history_rows']==300
  prov=json.loads((rd/'provenance.json').read_text());assert prov['seed']==42 and prov['parameters']==910627
  assert prov['source_lock']['models/MambaLiteUNet.py']==sha(ROOT/'references/code/MambaLiteUNet/models/MambaLiteUNet.py')
  assert prov['split_sha256']==c['split_sha256'] and prov['training']==c['training']
  m=api.MambaLiteUNet();m.load_state_dict(torch.load(fixed['checkpoint'],map_location='cpu',weights_only=False),strict=True)
  summary=json.loads((td/'test/summary.json').read_text());assert summary['checkpoint_sha256']==fixed['checkpoint_sha256'] and summary['split_sha256']==c['split_sha256']
  manifest=json.loads((td/'manifest.json').read_text());assert manifest['tta'] is False and manifest['threshold']==.5
  audit=independent_audit(td/'test',set(json.loads(Path(c['test_ids_file']).read_text())),summary)
  proof=dict(reusable=True,fixed=fixed,test_summary=str(td/'test/summary.json'),audit=audit)
 except Exception as e:proof=dict(reusable=False,error=repr(e),no_automatic_extra_training=True)
 save(LOG/'legacy_seed42_audit.json',proof)
 # Per-seed initialization and exact actual first augmentation batch, plus sampler reproducibility.
 states=[];signatures=[]
 for s in (42,43,44,45,46):
  api.seed_all(s);m=api.MambaLiteUNet()
  h=hashlib.sha256()
  for k,v in m.state_dict().items():h.update(k.encode());h.update(v.numpy().tobytes())
  states.append(h.hexdigest());batches=[]
  for repetition in range(2):
   api.seed_all(s);loaders,g=seed_loaders(s,workers=0);batch=next(iter(loaders['train']));batches.append(dict(ids=list(batch[2]),pixels=hashlib.sha256(batch[0].numpy().tobytes()+batch[1].numpy().tobytes()).hexdigest()));[v.close() for v in loaders.values() if hasattr(v,'close')]
  assert batches[0]==batches[1];signatures.append(batches[0])
  buffer=io.BytesIO();torch.save(m.state_dict(),buffer);buffer.seek(0);m.load_state_dict(torch.load(buffer,map_location='cpu',weights_only=False),strict=True)
 assert len(set(states))==5 and len({tuple(v['ids']) for v in signatures})==5 and len({v['pixels'] for v in signatures})==5
 # Existing validated reconstruction model/source reused; exact complete latest recovery tested on CPU.
 import reconstruction_engine as re
 from reconstruction_restore import restore_cpu
 re.verify_sources();m=re.build_model();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=50,eta_min=1e-5)
 loaders=re.make_loaders(42,workers=0);payload,audit=restore_cpu(m,opt,scheduler,loaders,42,re.run_path(42),re.model_path(42))
 save(LOG/'reconstruction_resume_acceptance.json',audit)
 assert not torch.cuda.is_initialized()
 save(LOG/'cpu_acceptance.json',dict(status='passed',CUDA_initialized=False,public_initial_hashes=dict(zip(range(42,47),states)),first_batches=dict(zip(range(42,47),signatures)),reconstruction=audit,legacy_seed42_reusable=proof['reusable'],scope='CPU strict model roundtrip, actual seeded loader batches and resume state; no full CPU public forward/backward; GPU real batch deferred',protocol_sha256=sha(OPS/'protocol.json')))
 print('CPU_ACCEPTANCE_PASSED',json.dumps(audit),flush=True)
def phase(mode,arm):
 c=verify();seed,rd,md,recon=identity(arm);rd.mkdir(parents=True,exist_ok=True);md.mkdir(parents=True,exist_ok=True)
 assert json.loads((LOG/'cpu_acceptance.json').read_text())['protocol_sha256']==sha(OPS/'protocol.json')
 api.seed_all(seed);torch.set_num_threads(4)
 if mode=='preflight':
  torch.cuda.set_device(0);device=torch.device('cuda:0');m=build(recon).to(device)
  if recon:
   import reconstruction_engine as re
   loaders=re.make_loaders(seed,workers=0)
  else:loaders,g=seed_loaders(seed,workers=0)
  opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8)
  out=rd/'preflight';out.mkdir(exist_ok=False)
  allowed=current_protocol(CONFIGS/'protocol_v5_dedup_seed42.json')['models']['mambalite']['inherited_unused_parameters']
  record=old.preflight(m,loaders,api.BceDiceLoss().to(device),opt,device,out,allowed)
  apps=os.popen('nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader').read()
  own=[x.split(',')[0].strip() for x in apps.splitlines() if x.split(',')[1].strip()==str(os.getpid())];assert own==[c['gpu_uuid']]
  record.update(gpu_uuid=c['gpu_uuid'],compute_pid=os.getpid(),protocol_sha256=sha(OPS/'protocol.json'));save(rd/'gpu_preflight.json',record)
  print('REAL_BATCH8_PREFLIGHT_PASSED',arm,json.dumps(record),flush=True);return
 assert json.loads((rd/'gpu_preflight.json').read_text())['status']=='passed'
 if mode=='train':
  if recon:
   import reconstruction_engine as re
   re.train(42);return
  assert not (rd/'run/history.csv').exists(),'fresh start refuses prior history'
  run=rd/'run';run.mkdir(exist_ok=False);(run/'checkpoints').symlink_to(md,target_is_directory=True)
  torch.cuda.set_device(0);device=torch.device('cuda:0');api.seed_all(seed);m=api.MambaLiteUNet().to(device)
  loaders,g=seed_loaders(seed);conf=current_protocol(CONFIGS/'protocol_v5_dedup_seed42.json')
  info=dict(model='mambalite',seed=seed,parameters=sum(p.numel() for p in m.parameters()),protocol_sha256=sha(OPS/'protocol.json'),split_sha256=c['split_sha256'],physical_gpu=1,initialization='fresh random independent seed',training=c['training'],source_sha256=c['sources'])
  h=hashlib.sha256()
  for k,v in m.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
  info['initial_state_sha256']=h.hexdigest()
  assert info['initial_state_sha256']==json.loads((LOG/'cpu_acceptance.json').read_text())['public_initial_hashes'][str(seed)]
  save(run/'provenance.json',info)
  import public_train
  public_train.SEED=seed
  public_train.train(m,loaders,api.BceDiceLoss().to(device),torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8),device,run,conf,info,g)
  assert ft.training_ready(run)
 elif mode=='test':
  if recon:
   import reconstruction_engine as re
   # CPU strict load is performed before CUDA initializes in this process.
   cp=md/'best.pth';m=build(True);m.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False),strict=True);assert not torch.cuda.is_initialized()
   save(rd/'cpu_strict_load.json',dict(status='passed',sha256=sha(cp),strict=True,CUDA_initialized=False))
   re.test(42)
   result=json.loads((rd/'test/summary.json').read_text());audit=independent_audit(rd/'test',set(json.loads(Path(c['test_ids_file']).read_text())),result);save(rd/'independent_final_audit.json',audit);return
  run=rd/'run';assert ft.training_ready(run);fixed=ft.fixed_best_record(run);assert fixed['history_rows']==300 and fixed['seed']==seed
  m=api.MambaLiteUNet();m.load_state_dict(torch.load(fixed['checkpoint'],map_location='cpu',weights_only=False),strict=True);assert not torch.cuda.is_initialized()
  save(rd/'cpu_strict_load.json',dict(strict=True,CUDA_initialized=False,**fixed))
  output=rd/'test';assert not output.exists();ds=api.HAMDataset(Path(c['split_file']).parent,'test');assert not ds.training
  loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=0,pin_memory=True)
  torch.cuda.set_device(0);device=torch.device('cuda:0');m=m.to(device).eval()
  save(rd/'test_manifest.json',dict(**fixed,tta=False,threshold=.5,test_ids_file=c['test_ids_file'],physical_gpu=1))
  print('FULL_TEST_START',arm,'images2020',flush=True)
  summary=api.evaluate(m,ft.ProgressLoader(loader,rd/'test_status.json',{'arm':arm}),api.BceDiceLoss().to(device),device,output)
  summary.update(**fixed,split='test',model='public_mambalite',tta=False,threshold=.5)
  assert sha(fixed['checkpoint'])==fixed['checkpoint_sha256'];verify()
  audit=independent_audit(output,set(json.loads(Path(c['test_ids_file']).read_text())),summary)
  save(output/'summary.json',summary);(output/'report.txt').write_text('Public MambaLite four-channel-branch code; HAM2020 noTTA; validation minimum loss best\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
  save(rd/'artifact_audit.json',audit);print('TEST_AUDIT_COMPLETE',arm,flush=True)
 else:raise ValueError(mode)
def aggregate():
 c=verify();legacy=json.loads((LOG/'legacy_seed42_audit.json').read_text());result=dict(public_seed42_reused=legacy['reusable'],reconstruction_in_public_statistics=False)
 public=[]
 if legacy['reusable']:public.append(json.loads((Path(c['legacy_seed42']['test_root'])/'test/summary.json').read_text()))
 for s in (43,44,45,46):
  rd=LOG/f'public{s}';assert json.loads((rd/'artifact_audit.json').read_text())['complete'];public.append(json.loads((rd/'test/summary.json').read_text()))
 result['public_runs']=len(public);result['public_summaries']=[dict(seed=r['seed'],global_metrics=r['global'],per_image_mean=r['per_image_mean'],checkpoint_sha256=r['checkpoint_sha256']) for r in public]
 if len(public)==5:
  result['five_seed_statistics']={section:{key:dict(mean=statistics.mean(r[section][key] for r in public),sample_std=statistics.stdev(r[section][key] for r in public),ddof=1) for key in keys} for section,keys in [('global',['dice','iou']),('per_image_mean',['dice','iou','hd95_pixels'])]}
 re=LOG/'fig2b/seed42';assert json.loads((re/'artifact_audit.json').read_text())['complete'];result['separate_reconstruction_seed42']=json.loads((re/'test/summary.json').read_text())
 save(LOG/'results.json',result);(LOG/'report.md').write_text('# MambaLite public four additions and separate Figure2b seed42\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['cpu','preflight','train','test','aggregate']);p.add_argument('--arm',choices=['public43','public44','public45','public46','fig2b42']);a=p.parse_args()
 # candidate_model import is required for reconstruction but never old controller.
 sys.path.append(str(ROOT/'source/mambalite_paper_candidate_20261105'))
 if a.mode=='cpu':cpu()
 elif a.mode=='aggregate':aggregate()
 else:phase(a.mode,a.arm)

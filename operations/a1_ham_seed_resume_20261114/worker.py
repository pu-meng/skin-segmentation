#!/usr/bin/env python3
"""Audit and resume two existing HAM seed experiments at epoch boundaries."""
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
OUT=Path('/media/8T/PuMengYu/experiments/skin_segmentation')
OPS=ROOT/'operations/a1_ham_seed_resume_20261114'
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
CASES={
 'esd40':{'root':OUT/'A1_ESD_SINGLE_HAM_ISIC_SEED40_FRESH300_20261114','work':'HAM10000/E+S+D2',
          'protocol':'protocol.json','seed':40,'epoch':26},
 'a143':{'root':OUT/'A1_SWIFTFORMER_HAM_SEED43_FRESH300_20261113','work':'swiftformer_s__A1',
         'protocol':'swiftformer_s__A1/protocol.json','seed':43,'epoch':252},
}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
 path=Path(path);tmp=path.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+'\n');tmp.replace(path)
def load_seed43_entry():
 directory=ROOT/'operations/a1_swiftformer_ham_seed43_20261113'
 sys.path.insert(0,str(directory));spec=importlib.util.spec_from_file_location('seed43_entry',directory/'entry.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
def build(name):
 if name=='esd40':
  from source.a1_swift_esd_single_20261113.candidate import build_candidate
  return build_candidate('E+S+D2',pretrained=False)
 from operations.encoder_swap_A016_20261107.encoder_swap_candidate import build_candidate
 return build_candidate('swiftformer_s__A1',pretrained=False)
def audit(name, require_epoch=True):
 import torch
 item=CASES[name]; root=item['root'];work=root/item['work'];run=work/'run'; protocol_path=root/item['protocol'];protocol=read(protocol_path)
 cfg=protocol.get('datasets',{}).get('HAM10000',{}).get('config') or protocol['config']
 payload=torch.load(run/'checkpoints/latest.pth',map_location='cpu',weights_only=False)
 rows=list(csv.DictReader((run/'history.csv').open())); epoch=int(payload['epoch'])
 assert (not require_epoch or epoch==item['epoch']) and [int(r['epoch']) for r in rows]==list(range(1,epoch+1))
 assert int(rows[-1]['optimizer_updates'])==epoch*cfg['training']['updates_per_epoch']
 assert payload['protocol_sha256']==sha(protocol_path) and payload['split_sha256']==cfg['data']['split_sha256']==sha(cfg['data']['split_file'])
 assert all(k in payload for k in ['model','optimizer','scheduler','python_rng','numpy_rng','torch_rng','cuda_rng','loader_rng'])
 best=min(rows,key=lambda row:float(row['val_loss']))
 assert int(best['epoch'])==payload['best_epoch'] and float(best['val_loss'])==payload['best_loss']
 assert (run/'checkpoints/best.pth').is_file() and not (work/'completion.json').exists()
 mismatches=[]
 for filename,expected in protocol['protected_sha256'].items():
  path=Path(filename)
  if not path.is_file() or sha(path)!=expected:mismatches.append(filename)
 if name=='esd40':
  altered=str(ROOT/'operations/a1_esd_single_seed40_20261114/entry.py')
  assert mismatches==[altered],mismatches
  snapshot=root/'source_snapshot'/Path(altered).relative_to('/')
  assert sha(snapshot)==protocol['protected_sha256'][altered]
  original=snapshot.read_text();current=Path(altered).read_text()
  assert original.replace('7168','4096')==current, 'entry drift extends beyond resource thresholds'
 else:
  assert not mismatches,mismatches
  load_seed43_entry().verify('swiftformer_s__A1')
 model=build(name);model.load_state_dict(payload['model'],strict=True)
 optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['training']['lr'],weight_decay=cfg['training']['weight_decay'],betas=tuple(cfg['training']['betas']),eps=cfg['training']['eps'])
 optimizer.load_state_dict(payload['optimizer'])
 scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=50,eta_min=1e-5);scheduler.load_state_dict(payload['scheduler'])
 lock=run/'RUNNING.lock'; stale_pid=None
 if lock.exists():
  stale_pid=int(lock.read_text().strip())
  if Path(f'/proc/{stale_pid}').exists():raise RuntimeError(f'run lock PID still exists: {stale_pid}')
 result={'case':name,'passed':True,'completed_epoch':epoch,'next_epoch':epoch+1,'checkpoint_sha256':sha(run/'checkpoints/latest.pth'),
         'protocol_sha256':sha(protocol_path),'split_sha256':payload['split_sha256'],
         'optimizer_updates':int(rows[-1]['optimizer_updates']),'best_epoch':payload['best_epoch'],
         'strict_model_optimizer_scheduler_load':True,'rng_fields_present':True,'stale_lock_pid':stale_pid,
         'source_exception':'seed40 entry changes only 7168->4096 GPU admission threshold' if mismatches else None}
 write(OPS/f'{name}_boundary_audit.json',result);print('BOUNDARY_AUDIT_PASS',json.dumps(result),flush=True)
 return cfg,payload,result

def resume(name,gpu):
 import torch,ham_fast_runner as runner,recover_ham_dedup
 from ham_fast_pipeline import make_fast_loaders
 cfg,payload,proof=audit(name);item=CASES[name];work=item['root']/item['work'];run=work/'run'
 lock=run/'RUNNING.lock'
 if lock.exists():
  pid=int(lock.read_text().strip())
  if Path(f'/proc/{pid}').exists():raise RuntimeError('existing running lock PID is alive')
  target=run/f'RUNNING.lock.stale_pid{pid}_20261114'
  if target.exists():raise FileExistsError(target)
  lock.rename(target)
 with lock.open('x') as f:f.write(str(os.getpid())+'\n')
 try:
  torch.set_num_threads(4);torch.cuda.set_device(0);api=runner.baseline_api();api.seed_all(item['seed'])
  model=build(name).cuda();loaders,generator=make_fast_loaders(cfg,api,workers=2,validation_batch=8)
  optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['training']['lr'],weight_decay=cfg['training']['weight_decay'],betas=tuple(cfg['training']['betas']),eps=cfg['training']['eps'])
  criterion=api.BceDiceLoss().cuda()
  old=read(run/'provenance.json')
  provenance=dict(old,seed=item['seed'],pid=os.getpid(),physical_gpu=gpu,fresh_start=False,
                  protocol_sha256=proof['protocol_sha256'],split_sha256=proof['split_sha256'],
                  recovery_source=str(OPS/'worker.py'),recovery_epoch=proof['completed_epoch'])
  write(OPS/f'{name}_resume_provenance.json',provenance)
  old_loaded=run/'resume_loaded.json'
  if old_loaded.exists():
   backup=OPS/f'{name}_previous_resume_loaded.json'
   if not backup.exists():shutil.copy2(old_loaded,backup)
  recover_ham_dedup.inspect_run=lambda model_name,config:dict(**proof,isolated_resume=True)
  try:runner.train(model,loaders,criterion,optimizer,torch.device('cuda:0'),run,cfg,provenance,generator,resume=True)
  finally:
   for loader in loaders.values():
    if hasattr(loader,'close'):loader.close()
  result=read(run/'artifact_audit.json')
  if not result.get('training_complete') or result.get('epochs')!=300:raise RuntimeError('training or validation incomplete')
  # The shared runner writes seed42 into every validation summary.
  validation=run/'validation/summary.json';summary=read(validation)
  original=OPS/f'{name}_validation_summary_before_seed_correction.json'
  if not original.exists():shutil.copy2(validation,original)
  summary['seed']=item['seed'];write(validation,summary)
  write(OPS/f'{name}_seed_metadata_audit.json',{'runner_seed':42,'correct_seed':item['seed'],
        'evidence':'original protocol and checkpoint; validation summary copy preserved', 'checkpoint_sha256':summary['checkpoint_sha256']})
 finally:
  if lock.exists() and lock.read_text().strip()==str(os.getpid()):lock.unlink()

def test(name,gpu):
 item=CASES[name]
 if name=='a143':
  module=load_seed43_entry();module.phase('swiftformer_s__A1','test',gpu)
  work=item['root']/item['work']
  for mode in ('none','mirror4'):
   p=work/'final_test'/mode/'summary.json';v=read(p);v['seed']=43;write(p,v)
  return
 from operations.a1_esd_single_seed40_20261114 import entry as e
 import torch,ham_fast_runner as runner
 model=build(name);api=runner.baseline_api();p=read(item['root']/'protocol.json')
 e.test('HAM10000','E+S+D2',gpu,p,model,torch,runner,api)

if __name__=='__main__':
 mode,name=sys.argv[1:3]
 if name not in CASES:raise ValueError(name)
 if mode=='audit':audit(name)
 elif mode=='resume':resume(name,int(sys.argv[3]))
 elif mode=='test':test(name,int(sys.argv[3]))
 else:raise ValueError(mode)

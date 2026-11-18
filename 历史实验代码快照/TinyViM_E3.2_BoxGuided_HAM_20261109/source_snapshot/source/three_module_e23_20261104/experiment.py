"""E1 only; immutable scientific protocol inherited from active four-stage E0."""
import os,sys,json,copy,shutil,signal
from pathlib import Path
import torch
from candidate import build_candidate,ROOT
sys.path.insert(0,str(ROOT/'source/bottleneck_mamba'))
import experiment_four_stage_decoder as base
import ham_fast_runner as runner
from ham_fast_pipeline import make_fast_loaders
from pretrained_unet2d import sha256_file
HERE=Path(__file__).parent
ARM=os.environ['E23_ARM']
RUN=ROOT/('log/TinyViM_三模块_'+ARM+('_HSA' if ARM=='E2' else '_FDCG')+'_HAM_20261104')
NAME=ARM

def prepare():
 config=copy.deepcopy(runner.validate_protocol(runner.PROTOCOL,'tinyvim_s',check_cache=True))
 old=json.loads((base.DEFAULT_ROOT/'protocol.json').read_text())
 for key in ['data','training','preprocessing','evaluation']:
  assert config[key]==old[key],key
 p=copy.deepcopy(old);p['version']=NAME;p['E0_reused_run']=str(base.DEFAULT_ROOT/'run')
 p['model']['new_module']=ARM+' only; E2 align bottleneck512x8x8 or E3 align highest skip64x64x64; pre conv -> mixer residual -> FFN residual -> post conv'
 p['model']['implementation_choices']=json.loads((HERE/'implementation_choices.json').read_text())
 p['source_sha256'].update({str(f.relative_to(ROOT)):sha256_file(f) for f in HERE.glob('*.py')})
 p['completion_contract']='300 epochs; min validation loss earliest best; strict fixed-best 2020-image test, predictions metrics report 12 visualizations'
 RUN.mkdir(exist_ok=True)
 path=RUN/'protocol.json'
 if path.exists():assert json.loads(path.read_text())==p,'protocol drift'
 else:runner.write_json(path,p)
 for rel,digest in p['source_sha256'].items():
  target=RUN/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True)
  if target.exists():assert sha256_file(target)==digest
  else:shutil.copyfile(ROOT/rel,target)
 return config

def main(mode):
 runner.verify_runtime=base.verify_v1_runtime
 runtime=base.verify_v1_runtime();config=prepare()
 if mode=='prepare':return
 assert json.loads((HERE/('cpu_'+ARM+'_acceptance.json')).read_text())['passed']
 assert json.loads((HERE/('gpu_'+ARM+'_acceptance.json')).read_text())['passed']
 stage=RUN/('preflight' if mode=='preflight' else 'run');stage.mkdir(exist_ok=True)
 if mode=='train':
  proof=json.loads((RUN/'preflight/preflight_status.json').read_text());assert proof['status']=='passed';assert proof['protocol_sha256']==sha256_file(RUN/'protocol.json')
  assert not any((stage/n).exists() for n in ['status.json','checkpoints','history.csv'])
 lock=RUN/'RUNNING.lock';lock.write_text(str(os.getpid())) if not lock.exists() else (_ for _ in ()).throw(RuntimeError('existing lock'))
 loaders={}
 try:
  api=runner.baseline_api();api.seed_all(42);torch.set_num_threads(4);torch.cuda.set_device(0);device=torch.device('cuda:0')
  model=build_candidate(pretrained=True,arm=ARM).to(device)
  loaders,generator=make_fast_loaders(config,api,workers=2,validation_batch=8)
  criterion=api.BceDiceLoss().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8)
  info=dict(model=NAME,seed=42,mode=mode,pid=os.getpid(),protocol_file=str(RUN/'protocol.json'),protocol_sha256=sha256_file(RUN/'protocol.json'),split_sha256=config['data']['split_sha256'],source_sha256=json.loads((RUN/'protocol.json').read_text())['source_sha256'],parameters=sum(p.numel() for p in model.parameters()),physical_gpu=int(os.environ['EXPERIMENT_PHYSICAL_GPU']),process_cuda_index=0,gpu_name=torch.cuda.get_device_name(),training=config['training'],fresh_start=True,test_data_accessed=False)
  runner.write_json(stage/'provenance.json',info);runner.write_json(stage/'weight_loading_audit.json',model.encoder.loading_audit)
  if mode=='preflight':
   report=runner.preflight(model,loaders,criterion,optimizer,device,stage,[]);report.update(protocol_sha256=sha256_file(RUN/'protocol.json'),split_sha256=config['data']['split_sha256']);runner.write_json(stage/'preflight_status.json',report);print('REAL_PREFLIGHT_PASS',json.dumps(report),flush=True)
  else:runner.train(model,loaders,criterion,optimizer,device,stage,config,info,generator,resume=False)
 except BaseException as e:runner.write_json(stage/'failure.json',dict(error=repr(e),pid=os.getpid()));raise
 finally:
  for l in loaders.values():
   if hasattr(l,'close'):l.close()
  lock.unlink()

if __name__=='__main__':main(sys.argv[1])

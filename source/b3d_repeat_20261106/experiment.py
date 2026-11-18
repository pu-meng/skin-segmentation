"""Independent four basic architecture controls under unchanged four-stage E0 protocol."""
import os,sys,json,copy,shutil,signal
from pathlib import Path
import torch
from source.b3d_repeat_20261106.candidate import ROOT,added_parameters
from source.b3d_repeat_20261106.selection import ARM,RUN,CHOICES
from source.b3d_repeat_20261106.candidate import build_candidate
sys.path.insert(0,str(ROOT/'source/bottleneck_mamba'))
import experiment_four_stage_decoder as base
import ham_fast_runner as runner
from ham_fast_pipeline import make_fast_loaders
from pretrained_unet2d import sha256_file
HERE=Path(__file__).parent
NAME=ARM

def prepare():
 config=copy.deepcopy(runner.validate_protocol(runner.PROTOCOL,'tinyvim_s',check_cache=True))
 old=json.loads((base.DEFAULT_ROOT/'protocol.json').read_text())
 for key in ['data','training','preprocessing','evaluation']:
  assert config[key]==old[key],key
 p=copy.deepcopy(old);p['version']=NAME;p['E0_reused_run']=str(base.DEFAULT_ROOT/'run')
 p['model']['deepest']=CHOICES['placement']
 p['model']['implementation_choices']=CHOICES
 p['model']['added_module_seed']=44
 p['model']['predecessor']='E0 four-stage baseline, isolated authorized basic architecture controls'
 p['model']['HAM_checkpoint_loaded']=False
 p['model']['note']='Official basic Mamba1 and/or official MedNeXt2d; no E1/E2/E3; no HAM weights'
 p['model']['initialization_seeds']=[42,43]
 p['source_sha256'].update({str(f.relative_to(ROOT)):sha256_file(f) for f in list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))})
 import inspect
 from mamba_ssm.modules.mamba_simple import Mamba
 p['official_mamba_source']={'path':inspect.getfile(Mamba),'sha256':sha256_file(inspect.getfile(Mamba))}
 for relative in ['Models/TinyViM/blocks.py','source/bd_controls_20261105/official_mednext.py']:
  p['source_sha256'][relative]=sha256_file(ROOT/relative)
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
 signal.signal(signal.SIGUSR1,runner.request_stop)
 if mode=='prepare':return
 proof=json.loads((RUN/'cpu_check.json').read_text());assert proof['passed']
 stage=RUN/('preflight' if mode=='preflight' else 'run');stage.mkdir(exist_ok=True)
 if mode=='train':
  proof=json.loads((RUN/'preflight/preflight_status.json').read_text());assert proof['status']=='passed';assert proof['protocol_sha256']==sha256_file(RUN/'protocol.json')
  assert not any((stage/n).exists() for n in ['status.json','checkpoints','history.csv'])
 lock=RUN/'RUNNING.lock'
 with lock.open('x') as f:f.write(str(os.getpid()))
 loaders={}
 try:
  api=runner.baseline_api();api.seed_all(42);torch.set_num_threads(4);torch.cuda.set_device(0);device=torch.device('cuda:0')
  model=build_candidate(pretrained=True).to(device)
  assert sum(p.numel() for p in model.parameters())==json.loads((RUN/'cpu_check.json').read_text())['parameters']
  msrc=json.loads((RUN/'protocol.json').read_text())['official_mamba_source'];assert sha256_file(msrc['path'])==msrc['sha256']
  for rel,digest in json.loads((RUN/'protocol.json').read_text())['source_sha256'].items():
   assert sha256_file(ROOT/rel)==digest, 'source drift: '+rel
  loaders,generator=make_fast_loaders(config,api,workers=2,validation_batch=8)
  criterion=api.BceDiceLoss().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8)
  info=dict(model=NAME,seed=42,mode=mode,pid=os.getpid(),protocol_file=str(RUN/'protocol.json'),protocol_sha256=sha256_file(RUN/'protocol.json'),split_sha256=config['data']['split_sha256'],source_sha256=json.loads((RUN/'protocol.json').read_text())['source_sha256'],parameters=sum(p.numel() for p in model.parameters()),physical_gpu=int(os.environ['EXPERIMENT_PHYSICAL_GPU']),process_cuda_index=0,gpu_name=torch.cuda.get_device_name(),training=config['training'],fresh_start=True,test_data_accessed=False)
  runner.write_json(stage/'provenance.json',info);runner.write_json(stage/'weight_loading_audit.json',model.encoder.loading_audit)
  if mode=='preflight':
   from Models.TinyViM.four_stage_decoder import build_four_stage_baseline
   baseline=build_four_stage_baseline(True).to(device).eval()
   disabled=build_candidate(True,count=0,med=False).to(device).eval()
   probe=torch.rand(1,3,256,256,device=device)
   with torch.no_grad():assert torch.equal(baseline(probe),disabled(probe)), 'GPU all-off mismatch'
   del baseline,disabled,probe
   torch.cuda.empty_cache()
   before={n:p.detach().clone() for n,p in added_parameters(model)}
   report=runner.preflight(model,loaders,criterion,optimizer,device,stage,[])
   unchanged=[n for n,p in added_parameters(model) if torch.equal(p.detach(),before[n])]
   assert not unchanged, 'unchanged inserted tensors: '+str(unchanged)
   report['all_added_parameter_tensors_updated']=True
   report.update(protocol_sha256=sha256_file(RUN/'protocol.json'),split_sha256=config['data']['split_sha256']);runner.write_json(stage/'preflight_status.json',report);print('REAL_PREFLIGHT_PASS',json.dumps(report),flush=True)
  else:runner.train(model,loaders,criterion,optimizer,device,stage,config,info,generator,resume=False)
 except BaseException as e:runner.write_json(stage/'failure.json',dict(error=repr(e),pid=os.getpid()));raise
 finally:
  for l in loaders.values():
   if hasattr(l,'close'):l.close()
  lock.unlink()

if __name__=='__main__':main(sys.argv[1])



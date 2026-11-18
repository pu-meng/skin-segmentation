import os,sys,json,copy,shutil,fcntl
from pathlib import Path
from candidate import ROOT,build_candidate,TrainingCriterion
import torch
import ham_fast_runner as runner
from ham_fast_pipeline import make_fast_loaders
from pretrained_unet2d import prepare_imports,sha256_file
prepare_imports('tinyvim_s');prepare_imports('mambavision_t')
from experiment_four_stage_decoder import verify_v1_runtime
HERE=Path(__file__).parent
RUN=ROOT/'log/TinyViM_E3.2_BoxGuided_HAM_20261109'
NAME='E3.2'
def prepare():
 RUN.mkdir(exist_ok=True)
 runner.verify_runtime=verify_v1_runtime
 audit=verify_v1_runtime();runner.write_json(RUN/'runtime_audit.json',audit)
 config=copy.deepcopy(runner.validate_protocol(runner.PROTOCOL,'tinyvim_s',check_cache=True))
 original=json.loads((ROOT/'log/TinyViM_三模块_E3_FDCG_HAM_20261104/protocol.json').read_text())
 for k in ['data','training','preprocessing','evaluation']:assert config[k]==original[k],k
 p=copy.deepcopy(original);p['version']=NAME;p['parent_E3_protocol']=str(ROOT/'log/TinyViM_三模块_E3_FDCG_HAM_20261104/protocol.json')
 p['model']['new_module']='Original E3 intact; D=ups0(D1) -> BoxHead -> soft W; Snew=S+W*(E3(S)-S); concat D,Snew -> original D0'
 p['box_implementation']=dict(status='first implementation choices fixed before test',head='Conv3 64to32 + GELU + spatial mean + Linear32to4; isolated seed44',coordinates='cxcywh; sigmoid width,height in (1/256,1); centres constrained inside width-dependent legal interval',soft_edges='product of four sigmoids; pixel centres; tau=2/64 normalized; no hard crop',supervision='current augmented train mask inclusive pixel bbox; empty GT omitted; singleton 1/256; boundary clipped by mask bounds',loss='train BCE+Dice + 0.1 * SmoothL1(cxcywh,beta=0.1), foreground samples only',selection='validation segmentation BCE+Dice only; aux omitted in eval',inference='predicted box only; never GT; no TTA')
 p['source_sha256'].update({str(f.relative_to(ROOT)):sha256_file(f) for f in HERE.glob('*.py')})
 # Pin all old training dependencies; no drift exemption and no old manifest edits.
 for rel,digest in original['source_sha256'].items():assert sha256_file(ROOT/rel)==digest,rel
 pp=RUN/'protocol.json'
 if pp.exists():assert json.loads(pp.read_text())==p,'protocol drift'
 else:runner.write_json(pp,p)
 for rel,digest in p['source_sha256'].items():
  target=RUN/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True)
  if target.exists():assert sha256_file(target)==digest
  else:shutil.copyfile(ROOT/rel,target)
 return config

def main(mode):
 config=prepare()
 if mode=='prepare':return
 cpu=json.loads((RUN/'cpu_acceptance.json').read_text());gpu=json.loads((RUN/'gpu_acceptance.json').read_text())
 assert cpu['passed'] and gpu['passed'] and gpu['protocol_sha256']==sha256_file(RUN/'protocol.json')
 stage=RUN/'run';stage.mkdir(exist_ok=True)
 assert not any((stage/n).exists() for n in ['status.json','history.csv','checkpoints'])
 lock=(RUN/'RUNNING.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);loaders={}
 try:
  api=runner.baseline_api();api.seed_all(42);torch.set_num_threads(4);torch.cuda.set_device(0);device=torch.device('cuda:0')
  model=build_candidate(pretrained=True).to(device)
  loaders,generator=make_fast_loaders(config,api,workers=2,validation_batch=8)
  criterion=TrainingCriterion(model,api.BceDiceLoss()).to(device)
  optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8)
  info=dict(model=NAME,seed=42,mode=mode,pid=os.getpid(),protocol_file=str(RUN/'protocol.json'),protocol_sha256=sha256_file(RUN/'protocol.json'),split_sha256=config['data']['split_sha256'],source_sha256=json.loads((RUN/'protocol.json').read_text())['source_sha256'],parameters=sum(p.numel() for p in model.parameters()),physical_gpu=int(os.environ['EXPERIMENT_PHYSICAL_GPU']),training=config['training'],fresh_start=True,test_data_accessed=False,initialization='official ImageNet encoder only; no trained HAM checkpoint; original E3 common tensors exact')
  runner.write_json(stage/'provenance.json',info);runner.write_json(stage/'weight_loading_audit.json',model.encoder.loading_audit)
  runner.train(model,loaders,criterion,optimizer,device,stage,config,info,generator,resume=False)
 except BaseException as e:runner.write_json(stage/'failure.json',dict(error=repr(e),pid=os.getpid()));raise
 finally:
  for l in loaders.values():
   if hasattr(l,'close'):l.close()
  lock.close()
if __name__=='__main__':main(sys.argv[1])

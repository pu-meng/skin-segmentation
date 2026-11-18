"""Isolated validation-only 100-epoch entry; never imports old experiment main."""
import os,sys,json,copy,shutil,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
from pretrained_unet2d import prepare_imports,sha256_file
prepare_imports('mambavision_t');prepare_imports('tinyvim_s')
import torch
from source.e123_candidate_a_20261106 import candidate
from source.e123_candidate_a_20261106.candidate import build_candidate
from source.bottleneck_mamba.experiment_v1 import verify_v1_runtime
import ham_fast_runner as runner
from ham_fast_pipeline import make_fast_loaders
RUN=ROOT/'log/TinyViM_E123_CandidateA_E2Spectral_B3x3_HAM100_20261106'
NAME='E123-CandidateA-E2Spectral-B3x3-HAM100'
def prepare():
 runner.verify_runtime=verify_v1_runtime
 runtime=verify_v1_runtime();config=copy.deepcopy(runner.validate_protocol(runner.PROTOCOL,'tinyvim_s',check_cache=True))
 oldpath=ROOT/'log/TinyViM_E123.1_联合新版_HAM_20261105/protocol.json';old=json.loads(oldpath.read_text())
 for k in ['data','training','preprocessing','evaluation']:assert config[k]==old[k],k
 config['training'].update(epochs=100,total_optimizer_updates=100*config['training']['updates_per_epoch'],image_exposures=100*config['data']['counts']['train'])
 paths=['source/e123_candidate_a_20261106/candidate.py','source/e123_candidate_a_20261106/entry.py','source/e123_candidate_a_20261106/check.py','source/e123_joint_20261105/candidate.py','source/three_module_impl_20261104/candidate.py','source/three_module_impl_20261104/official_wave.py','Models/TinyViM/blocks.py','Models/TinyViM/four_stage_decoder.py','source/comparison/ham_fast_runner.py','source/comparison/ham_fast_pipeline.py','source/comparison/pretrained_unet2d.py','source/bottleneck_mamba/experiment_v1.py']
 sources={p:sha256_file(ROOT/p) for p in paths}
 protocol=dict(version=NAME,parent_protocol=str(oldpath),parent_protocol_sha256=sha256_file(oldpath),data=config['data'],training=config['training'],preprocessing=config['preprocessing'],evaluation=config['evaluation'],source_sha256=sources,completion_contract='100 epochs; val minloss earliest strict tie; potential best batch1 recheck; final fixed-best val audit; NO TEST; NO 300 epoch continuation',initialization='same seed42 original joint construction; all retained E1/E3/public tensors paired; E2A isolated fork_rng seed44; ImageNet encoder only; no HAM/B3 trained weights',loader='original ham_fast_pipeline workers2; train batch8; fast val8, potential best and final val1',E2='3 independent official row-major one-way Mamba residuals d_model128 d_state16 d_conv4 expand2 dt_rank auto8; FFT channel statistics; one adaptive fusion; LN FFN256-512-256 GELU PW256-512; X+gamma output, gamma.01; no extra FFN residual',test_data_accessed=False)
 path=RUN/'protocol.json'
 if path.exists():assert json.loads(path.read_text())==protocol,'protocol drift'
 else:runner.write_json(path,protocol)
 for p,h in sources.items():
  dst=RUN/'source_snapshot'/p;dst.parent.mkdir(parents=True,exist_ok=True)
  if dst.exists():assert sha256_file(dst)==h
  else:shutil.copyfile(ROOT/p,dst)
 runner.write_json(RUN/'runtime_validation_audit.json',runtime)
 history=ROOT/'log/TinyViM_E123.1_联合新版_HAM_20261105/run/history.csv'
 rows=list(csv.DictReader(history.open()));prefix=rows[:100]
 assert [int(r['epoch']) for r in prefix]==list(range(1,101))
 best=min(prefix,key=lambda r:float(r['val_loss']))
 runner.write_json(RUN/'original_E123_first100_comparator.json',dict(reusable=True,history=str(history),history_sha256=sha256_file(history),compared_prefix_epochs=100,earliest_min_loss_row=best,NOT_original_300_best=True,protocol_equal_except_fixed_budget=True))
 return config

def main(mode):
 assert mode in ['prepare','preflight','train']
 assert Path(candidate.__file__).resolve()==ROOT/'source/e123_candidate_a_20261106/candidate.py'
 print('ENTRY_IDENTITY',__name__,__file__,'RUN',RUN,flush=True)
 config=prepare()
 if mode=='prepare':return
 assert json.loads((RUN/'cpu_acceptance.json').read_text())['passed']
 assert shutil.disk_usage(RUN).free>=8*2**30,'disk below 8GiB; no cleanup'
 assert torch.cuda.is_available(),'GPU required; no CPU fallback'
 stage=RUN/('preflight' if mode=='preflight' else 'run');stage.mkdir(exist_ok=True)
 if mode=='train':
  proof=json.loads((RUN/'preflight/preflight_status.json').read_text());assert proof['status']=='passed' and proof['protocol_sha256']==sha256_file(RUN/'protocol.json')
  assert not any((stage/f).exists() for f in ['status.json','history.csv','checkpoints'])
 loaders={}
 try:
  api=runner.baseline_api();api.seed_all(42);torch.set_num_threads(4);torch.cuda.set_device(0);device=torch.device('cuda:0')
  model=build_candidate(True).to(device)
  assert sum(p.numel() for p in model.parameters())==json.loads((RUN/'cpu_acceptance.json').read_text())['parameters']
  loaders,generator=make_fast_loaders(config,api,workers=2,validation_batch=8)
  criterion=api.BceDiceLoss().to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8)
  info=dict(model=NAME,seed=42,mode=mode,pid=os.getpid(),protocol_file=str(RUN/'protocol.json'),protocol_sha256=sha256_file(RUN/'protocol.json'),split_sha256=config['data']['split_sha256'],source_sha256=json.loads((RUN/'protocol.json').read_text())['source_sha256'],parameters=sum(p.numel() for p in model.parameters()),physical_gpu=int(os.environ['EXPERIMENT_PHYSICAL_GPU']),gpu_uuid=os.environ['EXPERIMENT_GPU_UUID'],process_cuda_index=0,training=config['training'],fresh_start=True,test_data_accessed=False)
  runner.write_json(stage/'provenance.json',info);runner.write_json(stage/'weight_loading_audit.json',model.encoder.loading_audit)
  if mode=='preflight':
   before={n:p.detach().clone() for n,p in model.decoder.e2.named_parameters()}
   proof=runner.preflight(model,loaders,criterion,opt,device,stage,[])
   unchanged=[n for n,p in model.decoder.e2.named_parameters() if torch.equal(before[n],p.detach())]
   assert not unchanged,'new unchanged tensors '+str(unchanged)
   proof.update(all_new_E2_tensors_updated=True,protocol_sha256=info['protocol_sha256'],split_sha256=info['split_sha256'],physical_gpu=info['physical_gpu'],gpu_uuid=info['gpu_uuid']);runner.write_json(stage/'preflight_status.json',proof)
   print('REAL_BATCH8_PREFLIGHT_PASS',json.dumps(proof),flush=True)
  else:
   runner.train(model,loaders,criterion,opt,device,stage,config,info,generator,resume=False)
   audit=json.loads((stage/'artifact_audit.json').read_text());assert audit['epochs']==100 and audit['training_complete'] and audit['validation_artifacts_complete'] and not audit['test_evaluation_complete']
 except BaseException as e:runner.write_json(stage/'failure.json',dict(error=repr(e),pid=os.getpid()));raise
 finally:
  for loader in loaders.values():
   if hasattr(loader,'close'):loader.close()
if __name__=='__main__':main(sys.argv[1])

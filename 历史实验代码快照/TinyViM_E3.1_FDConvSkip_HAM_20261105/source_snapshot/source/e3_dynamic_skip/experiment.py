import bootstrap
from bootstrap import ROOT
from pathlib import Path
import argparse,copy,fcntl,json,os,shutil,signal,sys,time,csv
import torch
from Models.E3DynamicSkip.model import build_candidate
from pretrained_unet2d import sha256_file
import ham_fast_runner as runner
from ham_fast_pipeline import make_fast_loaders
from experiment_v1 import verify_v1_runtime
from final_test_pipeline import training_ready,fixed_best_record,validate_outputs,ProgressLoader
HERE=Path(__file__).parent
OUT=ROOT/'log/TinyViM_E3.1_FDConvSkip_HAM_20261105'
E0=ROOT/'log/TinyViM_四层对应Decoder_基础_HAM_20261104/protocol.json'
LABEL='D4-E3.1 · Mamba＋RepDW7/FDConv跨层Skip'
def write(p,v):runner.write_json(p,v)
def sources():
 paths=set(json.loads(E0.read_text())['source_sha256'])
 paths.update(str(p.relative_to(ROOT)) for p in HERE.glob('*.py'))
 paths.update(str(p.relative_to(ROOT)) for p in (ROOT/'Models/E3DynamicSkip').rglob('*') if p.is_file() and '__pycache__' not in str(p))
 paths.update(['Models/E2Simplified/model.py','source/three_module_e23_20261104/candidate.py',
 'operations/nnunet_mirror_validation/mirror_core.py','operations/nnunet_mirror_validation/run_validation.py'])
 return {p:sha256_file(ROOT/p) for p in sorted(paths)}
def prepare():
 runner.verify_runtime=verify_v1_runtime;runtime=verify_v1_runtime()
 config=copy.deepcopy(runner.validate_protocol(runner.PROTOCOL,'tinyvim_s',check_cache=True));parent=json.loads(E0.read_text())
 for k in ['data','training','preprocessing','evaluation']:assert config[k]==parent[k],k
 config['models']['E3.1']=dict(run_dir=str(OUT/'run'),initialization='official ImageNet encoder; E0 paired fresh decoder42 and align43; new refiner42; no HAM segmentation weights')
 protocol=dict(version='E3.1_FDConvSkip_minimal_v1',parent_protocol=str(E0),parent_protocol_sha256=sha256_file(E0),
  data=config['data'],training=config['training'],preprocessing=config['preprocessing'],evaluation=config['evaluation'],
  source_sha256=sources(),seed=42,model=dict(insertion='S0 and D1-upsample output before original D0 concat',interface=['B',64,64,64],
   global_path='same S -> FullScan64: PW1+BN/DW3+SiLU/4-direction state16 rank4 scan/LN/PW1+BN',
   local_path='same S -> parallel full-width RepDW7 and official FDConv3; concat128 -> Conv1 64',
   gates='independent Gate(a,b)=a*(1+tanh(raw)*sigmoid(Conv1(b))); raw0; both read preupdate a/b',
   selection='Conv1(D,64->2) initialized zero; softmax along 2 branches; weights spatial B2HW broadcast channels',
   output='Snew=S+eta*(w0*Gate(A,B)+w1*Gate(B,A)+Conv1(D,64->64)); eta per-channel .001 nonzero',
   decoder='concat(original D,Snew) -> unchanged D0 -> original head/resize/sigmoid',
   FDConv=dict(kernel_num=4,kernel=3,padding=1,stride=1,groups=1,param_ratio=1,param_reduction=1,convert_param=True,
     FDW=True,KSM_Global=True,KSM_Local=True,FBM=True,FBM_k_list=[2,4,8],FBM_lowfreq_att=False,
     note='native FBM inside borrowed FDConv only; no external S high/low split or separate frequency main branch'),
   no_MPO=True,no_QKV=True,no_extra_FFN_pre_post_wrapper=True,all_parameters_trainable=True),
  borrowed_FDConv=json.loads((ROOT/'Models/E3DynamicSkip/vendor/origin.json').read_text()),
  authorization=dict(shared_GPU=True,protected_existing_processes='no modification or termination',OOM='block; no batch/size reduction or CPU fallback'),
  test=dict(selection='earliest minimum validation loss over 300 epochs',images=2020,modes=['none','mirror4'],
   primary='none vs E0 none; mirror4 supplementary vs E0 mirror4',views=[[],[2],[3],[2,3]],
   fusion='inverse flip actual head logits, mean, sigmoid; threshold >=.5',no_sliding_window=True,no_test_tuning=True,
   completion='fixed best SHA; strict CPU rebuild; predictions, per-image CSV, global Dice/IoU/accuracy/sensitivity/specificity, mean Dice/IoU/HD95 pixels, summary/report,12 visualization images per mode, artifact audit'))
 OUT.mkdir(exist_ok=True);p=OUT/'protocol.json'
 if p.exists():assert json.loads(p.read_text())==protocol,'protocol/source drift'
 else:write(p,protocol)
 for rel,digest in protocol['source_sha256'].items():
  dest=OUT/'source_snapshot'/rel
  if dest.exists():assert sha256_file(dest)==digest,rel
  else:dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
 write(OUT/'runtime_validation_audit.json',runtime)
 return config

def aggregate():
 result=json.loads((OUT/'results.json').read_text());s=result['results']['none'];m=s['per_image_mean']
 row=f"| {LABEL} | {m['dice']*100:.4f} | {m['iou']*100:.4f} | {m['hd95_pixels']:.4f} | {s['selected_epoch']} | {s['parameters']/1e6:.3f} | [无TTA summary](../{(OUT/'test/none/summary.json').relative_to(ROOT)})；[固定镜像对照](../{(OUT/'results.json').relative_to(ROOT)}) |"
 with (ROOT/'log/E3.1_report.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  path=ROOT/'研究记录/实验结果汇总.md';text=path.read_text();a=text.index('<a id="four-stage-ham"></a>');b=text.index('<a id="four-stage-isic"></a>',a);part=text[a:b];lines=part.splitlines();indices=[i for i,l in enumerate(lines) if l.startswith('| '+LABEL+' |')];assert len(indices)==1
  lines[indices[0]]=row;path.write_text(text[:a]+'\n'.join(lines)+'\n'+text[b:]);assert row in path.read_text()

def test(config):
 assert training_ready(OUT/'run');fixed=fixed_best_record(OUT/'run');assert fixed['history_rows']==300
 model=build_candidate(False);model.load_state_dict(torch.load(fixed['checkpoint'],map_location='cpu',weights_only=True),strict=True)
 assert not torch.cuda.is_initialized();write(OUT/'cpu_test_load_audit.json',dict(strict_load_passed=True,**fixed))
 api=runner.baseline_api();ds=api.HAMDataset(Path(config['data']['split_file']).parent,'test');assert len(ds)==2020 and not ds.training
 from torch.utils.data import DataLoader
 loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=2,pin_memory=True,multiprocessing_context='spawn')
 sys.path.insert(0,str(ROOT/'operations/nnunet_mirror_validation'))
 from run_validation import Predictor
 model.cuda().eval();results={}
 for mode in ['none','mirror4']:
  output=OUT/'test'/mode;assert not output.exists(),'Refuse partial test overwrite'
  pred=Predictor(model,mode=='mirror4').eval();state=dict(status='testing',mode=mode,pid=os.getpid(),**fixed)
  start=time.monotonic();s=api.evaluate(pred,ProgressLoader(loader,OUT/'test_status.json',state),api.BceDiceLoss().cuda(),torch.device('cuda'),output)
  assert pred.verify_remaining==0 and pred.max_error==0
  s.update(fixed);s.update(split='test',images=2020,model='E3.1',mode=mode,elapsed_seconds=time.monotonic()-start,test_used_for_checkpoint_selection=False,logits_hook_exact=True)
  assert sha256_file(fixed['checkpoint'])==fixed['checkpoint_sha256']
  write(output/'summary.json',s);write(output/'prediction_manifest.json',{p.name:sha256_file(p) for p in sorted((output/'predictions').glob('*.png'))})
  audit=validate_outputs(output,{r['image_id'] for r in ds.rows},s);write(output/'artifact_audit.json',audit)
  (output/'report.txt').write_text(f"E3.1 HAM test {mode},2020 images; fixed validation-best epoch {fixed['selected_epoch']}; SHA {fixed['checkpoint_sha256']}\nGlobal: {s['global']}\nPer-image mean: {s['per_image_mean']}\nHD95 pixels on256 grid. No test tuning.\n")
  results[mode]=s;pred.close()
 write(OUT/'results.json',dict(complete=True,results=results,primary='none; mirror4 supplementary',**fixed));write(OUT/'test_status.json',dict(status='complete',**fixed));aggregate()

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','preflight','train','test'],required=True);a=p.parse_args()
 config=prepare()
 if a.mode=='prepare':print('PREPARED',OUT/'protocol.json',flush=True);return
 with (OUT/'owner.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if a.mode=='test':test(config);return
  for kind in ['cpu','gpu']:assert json.loads((HERE/(kind+'_acceptance.json')).read_text())['passed']
  stage=OUT/('run' if a.mode=='train' else 'preflight');stage.mkdir(exist_ok=True)
  if a.mode=='train':
   proof=json.loads((OUT/'preflight/preflight_status.json').read_text());assert proof['status']=='passed' and proof['protocol_sha256']==sha256_file(OUT/'protocol.json')
   assert not any((stage/f).exists() for f in ['history.csv','checkpoints','status.json'])
  api=runner.baseline_api();api.seed_all(42);torch.set_num_threads(4);torch.cuda.set_device(0);device=torch.device('cuda:0')
  model=build_candidate(True).to(device)
  info=dict(model='E3.1',seed=42,pid=os.getpid(),fresh_start=True,test_data_accessed=False,parameters=sum(p.numel() for p in model.parameters()),physical_gpu=int(os.environ['EXPERIMENT_PHYSICAL_GPU']),process_cuda_index=0,protocol_file=str(OUT/'protocol.json'),protocol_sha256=sha256_file(OUT/'protocol.json'),split_sha256=config['data']['split_sha256'],training=config['training'],source_sha256=sources(),engineering=dict(workers=2,validation_batch=8,runner='unchanged ham_fast_runner',precision='FP32'))
  write(stage/'provenance.json',info);write(stage/'weight_loading_audit.json',model.encoder.loading_audit)
  loaders={}
  try:
   loaders,generator=make_fast_loaders(config,api,workers=2,validation_batch=8);crit=api.BceDiceLoss().to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8)
   signal.signal(signal.SIGUSR1,runner.request_stop)
   if a.mode=='preflight':
    report=runner.preflight(model,loaders,crit,opt,device,stage,[])
    grads={n:float(p.grad.double().norm()) for n,p in model.decoder.refiner.named_parameters()};assert all(v>0 for v in grads.values())
    # Independent test rebuild from actual short-run weights, before formal training.
    cp=stage/'diagnostic_test_rebuild.pth';torch.save(model.state_dict(),cp);other=build_candidate(False);other.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True),strict=True);cp.unlink();del other
    report.update(protocol_sha256=info['protocol_sha256'],all_new_grad_nonzero=True,gradient_norms=grads,independent_test_rebuild_strict=True,test_loader_constructed=False)
    write(stage/'preflight_status.json',report);print('REAL_PREFLIGHT_PASS',json.dumps(report),flush=True)
   else:runner.train(model,loaders,crit,opt,device,stage,config,info,generator,resume=False);assert training_ready(stage)
  except BaseException as e:write(stage/'failure.json',dict(mode=a.mode,error=repr(e),pid=os.getpid()));raise
  finally:
   for loader in loaders.values():
    if hasattr(loader,'close'):loader.close()
if __name__=='__main__':main()

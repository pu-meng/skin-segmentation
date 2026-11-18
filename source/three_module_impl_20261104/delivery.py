from pathlib import Path
R=Path('/home/PuMengYu/experiments/skin_segmentation');D=R/'source/three_module_impl_20261104'
(D/'acceptance.py').write_text('''import os,sys,json,io,time,importlib,hashlib
from pathlib import Path
MODE=sys.argv[1]
if MODE=='cpu':os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import torch
from candidate import build_candidate,DirectionalWaveOperator
from pretrained_unet2d import prepare_imports
prepare_imports('tinyvim_s');prepare_imports('mambavision_t')
torch.set_num_threads(1 if MODE=='cpu' else 4)
D=Path(__file__).parent
out={'device':MODE,'checks':{},'torch':torch.__version__}
def check(k,v):
 assert v,k
 out['checks'][k]=True;print('PASS',k,flush=True)
original=[]
if MODE=='cpu':
 from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
 class CPUReferenceScan:
  @staticmethod
  def apply(u,delta,A,B,C,D=None,delta_bias=None,delta_softplus=False,nrows=1):
   assert u.device.type=='cpu'
   return selective_scan_ref(u,delta,A,B,C,D,z=None,delta_bias=delta_bias,delta_softplus=delta_softplus)
 for name in ['Models.TinyViM.blocks','model.tvimblock']:
  m=importlib.import_module(name);original.append((m,m.SelectiveScan));m.SelectiveScan=CPUReferenceScan
try:
 torch.manual_seed(42)
 base=build_candidate(pretrained=True).eval();model=build_candidate(pretrained=True,d=True).eval()
 check('encoder_all_tensors_equal',all(torch.equal(v,model.encoder.state_dict()[k]) for k,v in base.encoder.state_dict().items()))
 out['loading_audit']=model.encoder.loading_audit
 check('encoder_official_loaded',out['loading_audit']['all_loaded_tensors_equal'])
 check('unmodified_tensors_equal',all(torch.equal(v,model.state_dict()[k]) for k,v in base.state_dict().items() if not k.startswith('decoder.stages.3.')))
 x=torch.rand(1,3,256,256)
 off=build_candidate(pretrained=False);off.load_state_dict(base.state_dict(),strict=True);off.eval()
 with torch.no_grad():check('all_off_bitwise',torch.equal(base(x),off(x)))
 del base,off
 device=torch.device('cpu' if MODE=='cpu' else 'cuda:0');model=model.to(device);model.train();x=x.to(device)
 if MODE!='cpu':
  torch.cuda.reset_peak_memory_stats();x=x.repeat(8,1,1,1)
 optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
 losses=[];grads={}
 for step in range(3):
  optimizer.zero_grad(set_to_none=True);y=model(x);check('full_shape_step'+str(step),tuple(y.shape)==(x.shape[0],1,256,256))
  target=torch.rand_like(y);loss=torch.nn.functional.binary_cross_entropy(y,target)+1-(2*(y*target).sum()+1)/(y.sum()+target.sum()+1)
  loss.backward();check('loss_finite_'+str(step),bool(torch.isfinite(loss)))
  check('all_trainable_grad_finite_'+str(step),all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad))
  grads={n:float(p.grad.double().norm()) for n,p in model.named_parameters() if n.startswith('decoder.stages.3.') and p.requires_grad}
  optimizer.step();losses.append(float(loss));print('OPTIMIZER_STEP',step,losses[-1],flush=True)
 check('all_new_tensor_grad_nonzero_step3',all(v>0 for v in grads.values()));out['new_parameter_gradient_norms']=grads;out['losses']=losses
 model.eval()
 with torch.no_grad():reference=model(x)
 buf=io.BytesIO();torch.save(model.state_dict(),buf);buf.seek(0)
 other=build_candidate(pretrained=False,d=True).to(device);other.load_state_dict(torch.load(buf,map_location=device,weights_only=True),strict=True);other.eval()
 with torch.no_grad():check('strict_roundtrip_bitwise',torch.equal(reference,other(x)))
 del other,optimizer
 op=model.decoder.stages[3].wave
 # Capture propagated spectra and independently verify rFFT validity.
 old=op._gate_and_project_out;specs=[]
 def capture(u,z,shape):
  specs.append(u.detach());return old(u,z,shape)
 op._gate_and_project_out=capture
 with torch.no_grad():
  for dims in [(8,8),(7,7),(7,8)]:
   xx=torch.randn(1,512,*dims,device=device);a=torch.randn(dims[0],dims[1]//2+1,1,device=device)
   yy=op(xx,a);sp=specs[-1];re=torch.fft.rfft2(torch.fft.irfft2(sp,s=dims,norm='ortho'),norm='ortho')
   check('conjugacy_'+str(dims),torch.allclose(re,sp,atol=2e-5,rtol=2e-5))
  check('zero_input_finite',bool(torch.isfinite(op(torch.zeros(1,512,8,8,device=device),torch.zeros(8,5,1,device=device))).all()))
 op._gate_and_project_out=old
 if MODE!='cpu':
  out['fp32_peak_allocated_bytes']=torch.cuda.max_memory_allocated();out['fp32_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
  model.train();model.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.float16):
   y=model(x);loss=y.float().square().mean()
  loss.backward();check('amp_finite',bool(torch.isfinite(loss)) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad))
  model.eval();model.zero_grad(set_to_none=True)
  with torch.no_grad():
   for _ in range(3):model(x)
   torch.cuda.synchronize();start=time.monotonic()
   for _ in range(10):model(x)
   torch.cuda.synchronize();out['latency_ms_batch8']=(time.monotonic()-start)*100
 out['parameters']=sum(p.numel() for p in model.parameters());out['flops']='not measured; FFT/complex/scan not counted'
finally:
 for m,orig in original:m.SelectiveScan=orig
out['cuda_initialized']=torch.cuda.is_initialized();out['passed']=True
(D/(MODE+'_acceptance.json')).write_text(json.dumps(out,ensure_ascii=False,indent=2))
print('ACCEPTANCE_PASS',MODE,len(out['checks']),flush=True)
''')
(D/'experiment.py').write_text('''"""E1 only; immutable scientific protocol inherited from active four-stage E0."""
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
RUN=ROOT/'log/TinyViM_三模块_E1_DWPO_HAM_20261104'
NAME='E1_DWPO'

def prepare():
 config=copy.deepcopy(runner.validate_protocol(runner.PROTOCOL,'tinyvim_s',check_cache=True))
 old=json.loads((base.DEFAULT_ROOT/'protocol.json').read_text())
 for key in ['data','training','preprocessing','evaluation']:
  assert config[key]==old[key],key
 p=copy.deepcopy(old);p['version']=NAME;p['E0_reused_run']=str(base.DEFAULT_ROOT/'run')
 p['model']['deepest']='replace entire D3 with official directional WaveOperator + gamma residual'
 p['model']['implementation_choices']=dict(heads=1,time=1.,speed='no separate speed in official source (implicit fixed1)',direction='ax=1+tanh(a), ay=1-tanh(a); init0',gamma='channelwise .001',optional_input_projection=False,alpha='official alpha_embed .05; conjugate boundary average',unmodified='DW3; Linear U0/V0/Z; alpha_ch; positive p; initial wave_weight0; LN/SiLU gate/output Linear')
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
 assert json.loads((HERE/'cpu_acceptance.json').read_text())['passed']
 assert json.loads((HERE/'gpu_acceptance.json').read_text())['passed']
 stage=RUN/('preflight' if mode=='preflight' else 'run');stage.mkdir(exist_ok=True)
 if mode=='train':
  proof=json.loads((RUN/'preflight/preflight_status.json').read_text());assert proof['status']=='passed';assert proof['protocol_sha256']==sha256_file(RUN/'protocol.json')
  assert not any((stage/n).exists() for n in ['status.json','checkpoints','history.csv'])
 lock=RUN/'RUNNING.lock';lock.write_text(str(os.getpid())) if not lock.exists() else (_ for _ in ()).throw(RuntimeError('existing lock'))
 loaders={}
 try:
  api=runner.baseline_api();api.seed_all(42);torch.set_num_threads(4);torch.cuda.set_device(0);device=torch.device('cuda:0')
  model=build_candidate(pretrained=True,d=True).to(device)
  loaders,generator=make_fast_loaders(config,api,workers=2,validation_batch=8)
  criterion=api.BceDiceLoss().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8)
  info=dict(model=NAME,seed=42,mode=mode,pid=os.getpid(),protocol_file=str(RUN/'protocol.json'),protocol_sha256=sha256_file(RUN/'protocol.json'),split_sha256=config['data']['split_sha256'],source_sha256=json.loads((RUN/'protocol.json').read_text())['source_sha256'],parameters=sum(p.numel() for p in model.parameters()),physical_gpu=1,process_cuda_index=0,gpu_name=torch.cuda.get_device_name(),training=config['training'],fresh_start=True,test_data_accessed=False)
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
''')
(D/'entry.py').write_text('''from pathlib import Path
import sys
from candidate import ROOT,build_candidate
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t');prepare_imports('tinyvim_s')
if __name__=='__main__':
 phase=sys.argv[1]
 if phase=='test':
  import final_test_pipeline as p
  from experiment import RUN
  p.ARMS={'E1':('TinyViM四级＋D-WPO',RUN/'run')};p.DEFAULT_ROOT=RUN/'final_test';p.build_model=lambda arm:build_candidate(pretrained=False,d=True)
  sys.argv=[sys.argv[0],'--arm','E1','--gpu','0'];p.main()
 else:
  from experiment import main
  main(phase)
''')
print('DELIVERED',D)

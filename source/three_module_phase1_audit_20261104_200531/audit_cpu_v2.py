"""CPU-only read audit. No optimizer, dataset, GPU, or candidate modules."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import sys,json,hashlib,traceback,ast,math,importlib,io,time
from functools import partial
R=Path('/home/PuMengYu/experiments/skin_segmentation')
E=Path('/home/PuMengYu/论文汇总/TinyViM三模块/evidence_20261104_200531')
sys.path[:0]=[str(R),str(R/'source/comparison')]
import torch
from torch import nn
from torch.nn import functional as F
torch.set_num_threads(1)
from pretrained_unet2d import prepare_imports,PretrainedUNet2D
prepare_imports('mambavision_t')
prepare_imports('tinyvim_s')
from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
from Models.TinyViM.model import TinyViMUNet
from Models.TinyViM.four_stage_decoder import build_four_stage_baseline
result={'device':'cpu','torch':torch.__version__,'checks':{},'not_implemented':['D-WPO','HSA-CG-B','FDCG-S']}
def save():
 (E/'cpu_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
def run(name,fn):
 try:
  record=fn();result['checks'][name]={'status':'PASS',**record};print(name,json.dumps(result['checks'][name],ensure_ascii=False),flush=True)
 except Exception as exc:
  result['checks'][name]={'status':'FAIL','error':repr(exc),'traceback':traceback.format_exc()};print(name,'FAIL',repr(exc),flush=True)
 save()
def gradients(model):
 rows={}
 for n,p in model.named_parameters():
  g=p.grad;rows[n]={'numel':p.numel(),'requires_grad':p.requires_grad,'has_grad':g is not None,'finite':bool(torch.isfinite(g).all()) if g is not None else False,'norm':float(g.double().norm()) if g is not None else None}
 return rows
class CPUReferenceScan:
 @staticmethod
 def apply(u,delta,A,B,C,D=None,delta_bias=None,delta_softplus=False,nrows=1):
  assert u.device.type=='cpu'
  return selective_scan_ref(u,delta,A,B,C,D,z=None,delta_bias=delta_bias,delta_softplus=delta_softplus)
modules=[importlib.import_module('Models.TinyViM.blocks'),importlib.import_module('model.tvimblock')]
original=[m.SelectiveScan for m in modules]
for m in modules:m.SelectiveScan=CPUReferenceScan
try:
 def pretrained_check():
  m=PretrainedUNet2D('tinyvim_s',pretrained=True);a=m.encoder.loading_audit
  (E/'pretrained_loading.json').write_text(json.dumps(a,ensure_ascii=False,indent=2))
  n=TinyViMUNet();n.encoder.backbone.load_state_dict(m.encoder.backbone.state_dict(),strict=True)
  equal=all(torch.equal(v,n.encoder.backbone.state_dict()[k]) for k,v in m.encoder.backbone.state_dict().items());assert equal
  return {'full_classifier_strict_load':a['full_classifier_strict_load'],'all_loaded_tensors_equal':a['all_loaded_tensors_equal'],'current_encoder_strict_transfer_equal':equal,'branch':a['checkpoint_branch'],'mapping_count':len(a['tensor_mapping'])}
 run('pretrained_loading',pretrained_check)
 def full_model(kind):
  torch.manual_seed(42)
  model=TinyViMUNet() if kind=='five_stage' else build_four_stage_baseline(pretrained=True)
  checkpoint_strict=None
  if kind=='five_stage':
   model.load_checkpoint(R/'log/八模型对比/runs/protocol_v5_dedup_seed42/tinyvim_s/seed42/checkpoints/best.pth');checkpoint_strict=True
  model.eval();x=torch.rand(1,3,256,256,requires_grad=True);shapes={};handles=[]
  def hook(name):
   def h(m,args,out):shapes[name]={'input':list(args[0].shape),'output':list(out.shape)}
   return h
  for i,a in enumerate(model.align):handles.append(a.register_forward_hook(hook('align'+str(i))))
  container=model.decoder.blocks if kind=='five_stage' else model.decoder.stages
  for i,b in enumerate(container):handles.append(b.register_forward_hook(hook('decoder'+str(i))))
  start=time.monotonic();y=model(x)
  for h in handles:h.remove()
  assert list(y.shape)==[1,1,256,256] and bool(torch.isfinite(y).all())
  loss=(y*torch.rand_like(y)).mean()+y.square().mean();loss.backward()
  grad=gradients(model);(E/(kind+'_gradients.json')).write_text(json.dumps(grad,indent=2))
  bad=[n for n,v in grad.items() if not v['has_grad'] or not v['finite']];zero=[n for n,v in grad.items() if v['norm']==0]
  assert not bad,bad
  assert bool(torch.isfinite(x.grad).all())
  with torch.no_grad():
   raw=model.raw_features(x.detach());aligned=[a(f) for a,f in zip(model.align,raw)];manual=model.decoder(aligned)
  # This is baseline forward decomposition, NOT three-switch-off equivalence.
  equal=torch.equal(y.detach(),manual);assert equal
  buff=io.BytesIO();torch.save(model.state_dict(),buff);buff.seek(0);state=torch.load(buff,map_location='cpu',weights_only=True);model.load_state_dict(state,strict=True)
  return {'input':[1,3,256,256],'output':list(y.shape),'raw_features':[list(f.shape) for f in raw],'interfaces':shapes,'parameters':sum(p.numel() for p in model.parameters()),'parameter_tensors':len(grad),'bad_gradients':bad,'zero_gradient_tensors':zero,'all_parameter_tensor_norms_positive':not zero,'input_gradient_finite':True,'baseline_forward_decomposition_bitwise_equal':equal,'strict_state_roundtrip':True,'segmentation_checkpoint_strict':checkpoint_strict,'diagnostic_elapsed_seconds':time.monotonic()-start}
 run('five_stage_cpu',lambda:full_model('five_stage'))
 run('four_stage_cpu',lambda:full_model('four_stage'))
finally:
 for m,o in zip(modules,original):m.SelectiveScan=o
 result['scan_restored']=all(m.SelectiveScan is o for m,o in zip(modules,original))

def load_ast(path,names,ns):
 tree=ast.parse(Path(path).read_text());nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names];assert {n.name for n in nodes}==set(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns);return ns

def official_fourier():
 ns=load_ast(R/'references/code/SU-RMT/Models/SURMT/HSAModule.py',['rotate_every_two','theta_shift','RoPE2D','FourierAttention'],{'torch':torch,'nn':nn,'F':F})
 torch.manual_seed(42);m=ns['FourierAttention'](4,64,64);shape=(1,4,64,64);q,k,v=[torch.randn(shape,requires_grad=True) for _ in range(3)];fw=torch.full((1,4,1),.5,requires_grad=True);rope=ns['RoPE2D'](256,4);sin,cos=rope(8,8);out=m(q,k,v,fw,8,8,sin,cos);assert list(out.shape)==list(shape)
 out.square().mean().backward();rows=gradients(m)
 assert all(r['has_grad'] and r['finite'] and r['norm']>0 for r in rows.values())
 assert all(t.grad is not None and bool(torch.isfinite(t.grad).all()) for t in [q,k,v,fw])
 return {'output':list(out.shape),'parameters':sum(p.numel() for p in m.parameters()),'gradients':rows,'inputs_grad_finite':True,'note':'exact official AST; heads=4, ratio=2 diagnostic only, not chosen HSA-CG-B config'}
run('official_fourier_attention',official_fourier)

def official_wave():
 from timm.models.layers import trunc_normal_
 p=E/'official_classification_models_waveformer.py'
 ns=load_ast(p,['_fft2d','_ifft2d','_fourier_lambda_map','_inv_softplus','WaveOperator'],{'torch':torch,'nn':nn,'F':F,'math':math,'trunc_normal_':trunc_normal_})
 torch.manual_seed(42);m=ns['WaveOperator'](res=8,dim=512,hidden_dim=512,num_heads=4);x=torch.randn(1,512,8,8,requires_grad=True);a=torch.full((8,5,4),-2.9706281,requires_grad=True);y=m(x,a);assert list(y.shape)==list(x.shape) and bool(torch.isfinite(y).all());y.square().mean().backward();rows=gradients(m);assert all(v['has_grad'] and v['finite'] for v in rows.values())
 (E/'official_wave_gradients.json').write_text(json.dumps(rows,indent=2))
 return {'output':list(y.shape),'parameters':sum(p.numel() for p in m.parameters()),'all_parameter_gradients_finite':True,'zero_gradient_tensors':[n for n,v in rows.items() if v['norm']==0],'alpha_input_gradient_finite':bool(torch.isfinite(a.grad).all()),'not_D_WPO':True,'note':'official initial wave_weight=0; projected v0 block of linear may have zero gradient; learnable time differs from requested fixed time'}
run('official_wave_operator',official_wave)
result['cuda_initialized']=torch.cuda.is_initialized();assert not result['cuda_initialized']
before=json.loads((E/'protected_before.json').read_text());after={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in before};(E/'protected_after.json').write_text(json.dumps(after,indent=2));changed=[n for n in before if before[n]!=after[n]]
result['protected_files']=len(before);result['protected_changed']=changed
result['passed_checks']=sum(v['status']=='PASS' for v in result['checks'].values());result['failed_checks']=sum(v['status']=='FAIL' for v in result['checks'].values())
save();print('AUDIT_SUMMARY',json.dumps({k:result[k] for k in ['passed_checks','failed_checks','cuda_initialized','scan_restored','protected_files','protected_changed']},ensure_ascii=False),flush=True)

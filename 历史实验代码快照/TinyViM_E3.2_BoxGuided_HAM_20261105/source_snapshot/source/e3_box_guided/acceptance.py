import os,sys,json,importlib,io,time
from pathlib import Path
MODE=sys.argv[1]
if MODE=='cpu':os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import torch
from candidate import build_candidate,old,auxiliary_box_loss,TrainingCriterion,soft_rectangle
from pretrained_unet2d import prepare_imports
prepare_imports('tinyvim_s');prepare_imports('mambavision_t')
from experiment import RUN,prepare
import ham_fast_runner as runner
D=Path(__file__).parent;torch.set_num_threads(1 if MODE=='cpu' else 4)
out={'device':MODE,'checks':{},'gradient_steps':[],'test_data_accessed':False};orig=[];loaders={}
def check(k,v):
 out['checks'][k]=bool(v)
 assert v,k
 print('PASS',k,flush=True)
if MODE=='cpu':
 from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
 class Reference:
  @staticmethod
  def apply(u,delta,A,B,C,D=None,delta_bias=None,delta_softplus=False,nrows=1):
   assert u.device.type=='cpu'
   return selective_scan_ref(u,delta,A,B,C,D,z=None,delta_bias=delta_bias,delta_softplus=delta_softplus)
 for name in ['Models.TinyViM.blocks','model.tvimblock']:
  m=importlib.import_module(name);orig.append((m,m.SelectiveScan));m.SelectiveScan=Reference
try:
 config=prepare();api=runner.baseline_api();api.seed_all(42)
 device=torch.device('cpu' if MODE=='cpu' else 'cuda:0')
 baseline=old.build_candidate(pretrained=True,arm='E3').eval()
 model=build_candidate(pretrained=True).eval()
 check('original_E3_common_initialization_exact',all(torch.equal(v,model.state_dict()[k]) for k,v in baseline.state_dict().items()))
 check('official_encoder',model.encoder.loading_audit['all_loaded_tensors_equal'])
 out['parameters_original_E3']=sum(p.numel() for p in baseline.parameters());out['parameters']=sum(p.numel() for p in model.parameters());out['added_parameters']=out['parameters']-out['parameters_original_E3']
 model=model.to(device);baseline=baseline.to(device);x=torch.rand(1,3,256,256,device=device)
 model.decoder.constraint_enabled=False
 with torch.no_grad():a=baseline(x);b=model(x)
 out['W1_max_abs_error']=float((a-b).abs().max());check('W1_original_output_equal',torch.equal(a,b))
 model.decoder.constraint_enabled=True;del baseline
 shapes={};hooks=[]
 for name,m in model.named_modules():
  if name in ['encoder','align','decoder.enhance','decoder.ups.0','decoder.stages.0','decoder.stages.1','decoder.stages.2','decoder.stages.3','decoder.box_head'] or name.startswith('align.'):
   def hook(mod,args,o,n=name):shapes[n]={'input':[list(z.shape) for z in args if torch.is_tensor(z)],'output':list(o.shape) if torch.is_tensor(o) else [list(z.shape) for z in o]}
   hooks.append(m.register_forward_hook(hook))
 if MODE!='cpu':
  from ham_fast_pipeline import make_fast_loaders
  loaders,generator=make_fast_loaders(config,api,workers=2,validation_batch=8);it=iter(loaders['train']);torch.cuda.reset_peak_memory_stats()
 model.train();seg=api.BceDiceLoss().to(device);criterion=TrainingCriterion(model,seg);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
 for step in range(3):
  if MODE=='cpu':
   x=torch.rand(1,3,256,256);y=torch.zeros(1,1,256,256);y[:,:,47:181,63:197]=1;ids=['synthetic']
  else:
   x,y,ids,_=next(it);x=x.to(device);y=y.to(device)
  opt.zero_grad(set_to_none=True);p=model(x);box=model.decoder.last_box
  check('output_shape_'+str(step),tuple(p.shape)==(len(x),1,256,256))
  lseg=seg(p,y);lbox=auxiliary_box_loss(box,y)
  coord=torch.autograd.grad(lseg,box,retain_graph=True)[0]
  check('segmentation_gradient_all_coordinates_'+str(step),bool(torch.isfinite(coord).all() and (coord.abs().sum(0)>0).all()))
  head=[q for n,q in model.named_parameters() if 'box_head' in n]
  auxgrad=torch.autograd.grad(lbox,head,retain_graph=True)
  check('auxiliary_loss_trains_head_'+str(step),all(torch.isfinite(g).all() and g.norm()>0 for g in auxgrad))
  loss=criterion(p,y);loss.backward()
  check('all_gradients_finite_'+str(step),all(q.grad is not None and torch.isfinite(q.grad).all() for q in model.parameters()))
  new={n:float(q.grad.double().norm()) for n,q in model.named_parameters() if 'box_head' in n}
  out['gradient_steps'].append({'step':step,'seg_loss':float(lseg.detach()),'box_loss':float(lbox.detach()),'coordinate_gradients':coord.detach().cpu().tolist(),'head_gradients':new,'image_ids':list(ids)})
  if step==2:check('all_original_E3_parameters_effective',all(q.grad.norm()>0 for n,q in model.named_parameters() if n.startswith('decoder.enhance.')))
  opt.step();print('OPTIMIZER_STEP',step,float(loss.detach()),flush=True)
 for h in hooks:h.remove()
 out['shapes']=shapes
 check('decoder_input_to_box',[*shapes['decoder.ups.0']['output'][1:]]==[64,64,64])
 # Exact algebra at the actual E3 interface; quantify inside/outside correction.
 s=torch.randn(1,64,64,64,device=device);e=model.decoder.enhance(s);box=torch.tensor([[.5,.5,.5,.5]],device=device)
 w=soft_rectangle(box,64,64);updated=s+w*(e-s);out['spatial']={'center_weight':float(w[0,0,32,32]),'corner_weight':float(w[0,0,0,0]),'center_correction':float((updated-s)[:,:,32,32].abs().mean()),'corner_correction':float((updated-s)[:,:,0,0].abs().mean())}
 check('outside_skip_preserved_softly',out['spatial']['corner_weight']<.001 and out['spatial']['center_weight']>.9)
 model.eval()
 with torch.no_grad():expected=model(x)
 # This is always a fresh CPU strict reconstruction, including GPU preflight.
 buf=io.BytesIO();torch.save(model.state_dict(),buf);buf.seek(0)
 other=build_candidate(pretrained=False);other.load_state_dict(torch.load(buf,map_location='cpu',weights_only=True),strict=True);other=other.to(device).eval()
 with torch.no_grad():actual=other(x)
 check('strict_roundtrip',torch.equal(expected,actual))
 if MODE!='cpu':
  val=api.evaluate(model,torch.utils.data.DataLoader(torch.utils.data.Subset(loaders['val'].dataset,[0,1]),batch_size=1,num_workers=0),seg,device)
  check('real_validation',val['images']==2);out['validation']=val
  out.update(peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,formal_batch=8,real_augmented_training=True)
 out.update(passed=True,preflight_updates_discarded=True,protocol_sha256=runner.sha256_file(RUN/'protocol.json'))
except BaseException as exc:
 out.update(passed=False,error=repr(exc));raise
finally:
 for m,v in orig:m.SelectiveScan=v
 for l in loaders.values():
  if hasattr(l,'close'):l.close()
 out.update(scan_reference_cpu_only=MODE=='cpu',scan_restored=all(m.SelectiveScan is v for m,v in orig),cuda_initialized=torch.cuda.is_initialized())
 runner.write_json(RUN/(MODE+'_acceptance.json'),out)
print('ACCEPTANCE_PASS',MODE,flush=True)

import os,sys,time,json,hashlib,traceback,types,inspect,ast,textwrap,subprocess
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
os.environ.update(CUDA_VISIBLE_DEVICES='1',MKL_THREADING_LAYER='GNU',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TRITON_CACHE_DIR=str(OUT/'triton_cache'))
sys.path[:0]=[str(OUT),str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
import numpy as np,torch,triton,random,copy
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t');prepare_imports('tinyvim_s')
from Models.TinyViM import blocks as b
from Models.E3DynamicSkip.model import build_candidate
import official_csm_triton as official
import ham_fast_runner as runner
from ham_fast_pipeline import PlannedDataset
from project_paths import current_protocol
from torch.utils.data import DataLoader
R=dict(commit='2ed52ead062a51a64521ed3871d52914bf532876',torch=torch.__version__,triton=triton.__version__,cuda=torch.version.cuda,pid=os.getpid(),gpu_uuid=subprocess.getoutput('nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader'),source_sha256=hashlib.sha256((OUT/'official_csm_triton.py').read_bytes()).hexdigest(),ops=[],fullmodel=[],windows=[],test_accessed=False,scope='isolated process, instance-bound TinyViM forward_core plus refiner FullScan.forward; original classes/globals untouched')
def save():
 t=OUT/'triton_report.tmp.json';t.write_text(json.dumps(R,indent=2));t.replace(OUT/'triton_report.json')
def compare(a,z,atol=1e-6,rtol=1e-4):
 info={}
 for n,x in a.items():
  y=z[n];d=x.double()-y.double();info[n]=dict(max_abs=float(d.abs().max()),relative_l2=float(d.norm()/(x.double().norm()+1e-12)),close=torch.allclose(x,y,atol=atol,rtol=rtol))
 return dict(passed=all(v['close'] for v in info.values()),max_abs=max(v['max_abs'] for v in info.values()),max_relative_l2=max(v['relative_l2'] for v in info.values()),per_tensor=info)
class CS:
 @staticmethod
 def apply(x):return official.cross_scan_fn(x,in_channel_first=True,out_channel_first=True,one_by_one=False,scans=0)
class CM:
 @staticmethod
 def apply(x):return official.cross_merge_fn(x,in_channel_first=True,out_channel_first=True,one_by_one=False,scans=0)
def main():
 torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.28);torch.manual_seed(42)
 save()
 # Genuine official API before any compatibility edits: rectangular, square, noncontiguous.
 for shape,noncontig in [((2,7,13,19),False),((8,64,64,64),False),((2,7,13,19),True)]:
  x=torch.randn(*shape,device='cuda');x=x.transpose(-1,-2) if noncontig else x;x.requires_grad_();z=x.detach().clone().requires_grad_()
  u=b.CrossScan.apply(x);v=CS.apply(z);grad=torch.randn_like(u);u.backward(grad);v.backward(grad)
  scan=dict(output=compare({'o':u},{'o':v}),gradient=compare({'g':x.grad},{'g':z.grad}))
  a=torch.randn(shape[0],4,shape[1],shape[2],shape[3],device='cuda',requires_grad=True);c=a.detach().clone().requires_grad_()
  u=b.CrossMerge.apply(a);v=CM.apply(c);grad=torch.randn_like(u);u.backward(grad);v.backward(grad)
  merge=dict(output=compare({'o':u},{'o':v}),gradient=compare({'g':a.grad},{'g':c.grad}))
  R['ops'].append(dict(shape=shape,noncontiguous=noncontig,scan=scan,merge=merge));save()
 # Instance-only copied function globals; preserve entire scan and pooling calculations.
 scan_global=dict(b.cross_selective_scan.__globals__,CrossScan=CS,CrossMerge=CM)
 scan=types.FunctionType(b.cross_selective_scan.__code__,scan_global,b.cross_selective_scan.__name__,b.cross_selective_scan.__defaults__)
 model=build_candidate(False).cuda();initial=torch.load(OUT/'initial_state.pth',map_location='cpu',weights_only=True)
 orig=[]
 for name,m in model.named_modules():
  if isinstance(m,b.SS2D):
   fn=m.forward_core.__func__;new=types.FunctionType(fn.__code__,dict(fn.__globals__,cross_selective_scan=scan),fn.__name__,fn.__defaults__);orig.append((m,'forward_core',m.forward_core,types.MethodType(new,m)))
 # Change only the refiner FullScan instance's call target in copied forward AST.
 m=model.decoder.refiner.global_path;fn=m.forward.__func__;tree=ast.parse(textwrap.dedent(inspect.getsource(fn)))
 class Replace(ast.NodeTransformer):
  def visit_Attribute(self,node):
   self.generic_visit(node)
   if isinstance(node.value,ast.Name) and node.value.id=='b' and node.attr=='cross_selective_scan':return ast.copy_location(ast.Name(id='_isolated_scan',ctx=ast.Load()),node)
   return node
 tree=ast.fix_missing_locations(Replace().visit(tree));g=dict(fn.__globals__,_isolated_scan=scan);exec(compile(tree,'<isolated_FullScan_calltarget>','exec'),g);orig.append((m,'forward',m.forward,types.MethodType(g['forward'],m)))
 R['bound_instances']=len(orig);save()
 def mode(triton_mode):
  for m,n,old,new in orig:setattr(m,n,new if triton_mode else old)
 api=runner.baseline_api();cfg=current_protocol(ROOT/'source/config/protocol_v5_dedup_seed42.json');ds=PlannedDataset(Path(cfg['data']['split_file']).parent,'train');plans=json.loads((OUT/'epoch_plan.json').read_text())['plans']
 # Real full-sized first3 fixed batches. Data/augmentation precomputed outside short window.
 loader=DataLoader(ds,batch_size=8,sampler=plans[:24],num_workers=0,pin_memory=True,drop_last=False)
 batches=list(loader);crit=api.BceDiceLoss().cuda();api.seed_all(42);rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state())
 def reset(triton_mode):
  mode(triton_mode);model.load_state_dict(initial);model.train();random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2]);torch.cuda.set_rng_state(rng[3]);return torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8)
 reference=[]
 for label,tri in [('baselineA',False),('baselineB',False),('triton',True)]:
  opt=reset(tri);rows=[]
  for step,(x,y,ids,_) in enumerate(batches,1):
   opt.zero_grad(set_to_none=True);p=model(x.cuda(non_blocking=True));loss=crit(p,y.cuda(non_blocking=True));loss.backward()
   grads={n:(p.grad.detach().cpu().clone() if p.grad is not None else torch.zeros_like(p,device='cpu')) for n,p in model.named_parameters()}
   opt.step();state={n:p.detach().cpu().clone() for n,p in model.state_dict().items()}
   item=dict(loss=float(loss),prediction=p.detach().cpu(),gradients=grads,state=state)
   if label=='baselineA':reference.append(item);rows.append(dict(step=step,loss=item['loss']))
   else:
    ref=reference[step-1];rows.append(dict(step=step,loss_abs=abs(item['loss']-ref['loss']),prediction=compare({'p':ref['prediction']},{'p':item['prediction']}),gradients=compare(ref['gradients'],grads),state=compare(ref['state'],state)))
  R['fullmodel'].append(dict(mode=label,steps=rows));save();print('NUMERICAL COMPLETE',label,flush=True)
 del reference,item,grads,state;torch.cuda.empty_cache()
 # Actual full-network train updates ABBA, repeated same three real batches, no per-step synchronize.
 # Not a full epoch and not a substitute for train+validation/save wall clock.
 for label,tri in [('eager',False),('triton',True),('triton',True),('eager',False)]:
  opt=reset(tri)
  for x,y,_,_ in batches:
   opt.zero_grad(set_to_none=True);crit(model(x.cuda(non_blocking=True)),y.cuda(non_blocking=True)).backward();opt.step()
  opt=reset(tri);torch.cuda.synchronize();t=time.perf_counter();losses=[]
  for step in range(60):
   x,y,_,_=batches[step%3];opt.zero_grad(set_to_none=True);loss=crit(model(x.cuda(non_blocking=True)),y.cuda(non_blocking=True));loss.backward();opt.step();losses.append(loss.detach())
  torch.cuda.synchronize();seconds=time.perf_counter()-t
  R['windows'].append(dict(mode=label,steps=60,seconds=seconds,loss=float(torch.stack(losses).mean()),finite=bool(torch.isfinite(torch.stack(losses)).all())));save();print('SHORT WINDOW',R['windows'][-1],flush=True)
 R['complete']=True;save();print('TRITON PROBE COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:R.update(complete=False,error=repr(e),traceback=traceback.format_exc());save();raise

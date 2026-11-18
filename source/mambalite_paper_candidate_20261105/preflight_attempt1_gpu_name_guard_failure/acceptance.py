import os,sys,json,time,io
from pathlib import Path
mode=sys.argv[1]
if mode=='cpu':os.environ['CUDA_VISIBLE_DEVICES']=''
else:os.environ['CUDA_VISIBLE_DEVICES']='1'
HERE=Path(__file__).resolve().parent
assert (HERE/'experiment.py').exists(), 'Candidate training/testing entry not implemented'
from experiment import *
api.seed_all(31415)
torch.set_num_threads(1 if mode=='cpu' else 4)
started=time.monotonic()
report={'scope':'full candidate network '+mode,'status':'running','pid':os.getpid(),'checks':[]}
def check(name,ok):
 report['checks'].append({'name':name,'passed':bool(ok)})
 assert ok,name
try:
 device=torch.device('cpu' if mode=='cpu' else 'cuda:0')
 m=build_model(cpu_reference=mode=='cpu').to(device)
 report['parameters']=sum(p.numel() for p in m.parameters())
 report['SS2D_blocks']=sum(isinstance(op,SS2D) for op in m.modules())
 check('56_SS2D_blocks',report['SS2D_blocks']==56)
 scales={n:float(p.detach()) for n,p in m.named_parameters() if n.endswith('.skip_scale')}
 check('only_six_AMF_alpha0',len(scales)==6 and all(v==0 for v in scales.values()))
 check('SSM_D_ones_not_zeroed',all(torch.equal(op.Ds,torch.ones_like(op.Ds)) for op in m.modules() if isinstance(op,SS2D)))
 criterion=api.BceDiceLoss().to(device)
 opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01)
 if mode=='cpu':
  batches=[(torch.randn(2,3,64,64),torch.randint(0,2,(2,1,64,64)).float(),[],[])]
 else:
  check('physical_GPU1_mapping',torch.cuda.get_device_name(0).endswith('4090') and os.environ['CUDA_VISIBLE_DEVICES']=='1')
  report['gpu_before']=gpu_snapshot()
  free,total=torch.cuda.mem_get_info()
  check('at_least_2GiB_free_before_preflight',free>2*2**30)
  torch.cuda.reset_peak_memory_stats()
  loaders=make_loaders(31415,workers=0)
  it=iter(loaders['train']);batches=[next(it) for _ in range(3)]
 losses=[]
 for i,(x,y,ids,_) in enumerate(batches):
  m.train();opt.zero_grad(set_to_none=True)
  pred=m(x.to(device));loss=criterion(pred,y.to(device))
  check(f'forward_{i}_shape_finite',pred.shape==y.shape and bool(torch.isfinite(pred).all()) and bool(torch.isfinite(loss)))
  loss.backward()
  missing=[n for n,p in m.named_parameters() if p.grad is None]
  expected=[n for n,p in m.named_parameters() if '.local_global_mixing.dwconv.' in n and n.rsplit('.',1)[-1] in ('weight','bias') and '.dwconv.dwconv.' not in n]
  check(f'backward_{i}_only_inherited_unused',set(missing)==set(expected) and len(expected)==12)
  check(f'backward_{i}_all_present_gradients_finite',all(bool(torch.isfinite(p.grad).all()) for p in m.parameters() if p.grad is not None))
  before=m.final.weight.detach().clone();opt.step()
  check(f'optimizer_{i}_changed_weights',not torch.equal(before,m.final.weight))
  losses.append(float(loss.detach()));print('ACCEPTANCE',mode,'optimizer step',i+1,'loss',losses[-1],flush=True)
 report['losses']=losses;report['inherited_unused_parameters']=missing
 m.eval()
 if mode=='gpu':
  val=next(iter(loaders['val']));x,y=val[0],val[1]
  report['val_image_ids']=list(val[2])
 else:x,y=batches[0][0],batches[0][1]
 with torch.no_grad():
  ref=m(x.to(device)).cpu();vloss=criterion(ref,y)
 check('validation_finite',bool(torch.isfinite(vloss)))
 buffer=io.BytesIO();torch.save(m.state_dict(),buffer);buffer.seek(0)
 loaded=build_model(cpu_reference=mode=='cpu').to(device)
 loaded.load_state_dict(torch.load(buffer,map_location=device),strict=True);loaded.eval()
 with torch.no_grad():actual=loaded(x.to(device)).cpu()
 check('strict_save_reload_same_outputs',torch.allclose(ref,actual,rtol=1e-5,atol=1e-6))
 report['validation_loss']=float(vloss)
 if mode=='cpu':check('CUDA_never_initialized',not torch.cuda.is_initialized())
 else:
  torch.cuda.synchronize();report.update(peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,gpu_after=gpu_snapshot())
 check('seed_specific_loader_streams',sampler_signature(42)!=sampler_signature(43) and sampler_signature(42)==sampler_signature(42))
 frozen=verify_sources()
 report['source_hashes']=frozen['source_hashes']
 check('frozen_sources_and_split_unchanged',True)
 report.update(status='passed',preflight_weights_discarded=True,elapsed_seconds=time.monotonic()-started)
except BaseException as e:
 import traceback
 report.update(status='failed',error=repr(e),traceback=traceback.format_exc())
 raise
finally:
 report['passed']=sum(c['passed'] for c in report['checks']);report['failed']=sum(not c['passed'] for c in report['checks'])
 save(HERE/(mode+'_acceptance.json'),report)
 print(json.dumps(report,ensure_ascii=False),flush=True)

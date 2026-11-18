import os,sys,json,fcntl,subprocess,time,traceback,importlib,csv,hashlib
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
OUT=Path('/media/8T/PuMengYu/experiments/skin_segmentation/ABLATION_REPAIR_FRESH300_20261112')
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba'),str(ROOT/'source/isic2018')]
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['EXPERIMENT_PHYSICAL_GPU']='1'
import torch
import ham_fast_runner as parent
from pretrained_unet2d import prepare_imports
import data
from final_test_pipeline import fixed_best_record,validate_outputs,ProgressLoader,training_ready
from torch.utils.data import DataLoader,Subset
ARMS=[(f,v) for f in ('swift','a1s1') for v in ('no_skip1','no_e2','no_d2d3')]
MOD={'swift':'source.a1_swift_ablate_20261111.candidate','a1s1':'source.a1s1_skipv06_ablate_unet_20261111.candidate'}
BASE=Path('/media/8T/PuMengYu/experiments/skin_segmentation/ISIC2018_A0_A1_FRESH300_20261108/protocol.json')
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str))
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def config(): return json.loads(BASE.read_text())['config']
def model(f,v,pretrained):
 prepare_imports('mambavision_t');prepare_imports('tinyvim_s')
 return importlib.import_module(MOD[f]).build_candidate(pretrained=pretrained,variant=v)
def close(ls):
 for l in ls.values():
  if hasattr(l,'close'):l.close()
def run(f,v,phase):
 assert (f,v) in ARMS
 c=config(); work=OUT/(f+'__'+v); work.mkdir(parents=True,exist_ok=True)
 api=parent.baseline_api();api.seed_all(42);torch.set_num_threads(4)
 m=model(f,v,phase!='test'); d=m.decoder.base if f=='a1s1' else m.decoder
 assert (d.e3 is None)==(v=='no_skip1') and (d.e2 is None)==(v=='no_e2')
 if v=='no_d2d3':assert all(type(d.stages[i]).__name__=='DoubleConv' for i in (2,3))
 device=torch.device('cuda:0');name=f+'__'+v
 if phase=='test':
  stage=work/'run'; assert training_ready(stage)
  fixed=fixed_best_record(stage);m.load_state_dict(torch.load(fixed['checkpoint'],map_location='cpu'),strict=True)
  write(work/'cpu_test_load_audit.json',{'strict_load_passed':True,'checkpoint_sha256':fixed['checkpoint_sha256'],'cuda_initialized':torch.cuda.is_initialized()})
  ds=data.ISICDataset(data.CACHE,'test');assert len(ds)==518 and not ds.training
  loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=2,pin_memory=True,multiprocessing_context='spawn')
  output=work/'test';assert not output.exists();m=m.to(device).eval(); criterion=api.BceDiceLoss().to(device)
  state={'arm':name,'status':'testing','total_images':518,**fixed}
  result=api.evaluate(m,ProgressLoader(loader,work/'test_status.json',state),criterion,device,output)
  result.update(dataset='ISIC2018',split='test',model=name,**fixed);write(output/'summary.json',result)
  write(work/'test_artifact_audit.json',validate_outputs(output,{r['image_id'] for r in ds.rows},result))
  write(work/'test_status.json',dict(state,status='test_complete',completed_images=518));print('TEST_COMPLETE',name,flush=True);return
 ls,g=data.loaders(c,api);assert len(ls['train'])==223 and len(ls['val'].dataset)==258 and len(ls['train'].dataset)==1780
 m=m.to(device); criterion=api.BceDiceLoss().to(device);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8)
 try:
  if phase=='preflight':
   m.train();x,y,ids,_=next(iter(ls['train']));assert tuple(x.shape)==(8,3,256,256)
   p=next(p for p in m.parameters() if p.requires_grad);before=p.detach().clone();opt.zero_grad(set_to_none=True)
   pred=m(x.to(device));loss=criterion(pred,y.to(device));assert torch.isfinite(loss);loss.backward()
   grads=[p.grad for p in m.parameters() if p.grad is not None];assert grads and all(bool(torch.isfinite(g).all()) for g in grads)
   opt.step();torch.cuda.synchronize();assert not torch.equal(before,p.detach())
   val=api.evaluate(m,DataLoader(Subset(ls['val'].dataset,[0,1]),batch_size=1,num_workers=0),criterion,device)
   write(work/'preflight.json',{'passed':True,'loss':float(loss),'val_images':val['images'],'peak_reserved':torch.cuda.max_memory_reserved(),'batch_shape':list(x.shape),'pid':os.getpid(),'physical_gpu':1})
   print('PREFLIGHT_PASSED',name,float(loss),flush=True);return
  assert json.loads((work/'preflight.json').read_text())['passed']
  stage=work/'run';stage.mkdir(exist_ok=False)
  prov={'model':name,'protocol_sha256':sha(OUT/'protocol.json'),'engineering':{'runner':'ham_fast_runner.train','workers':2,'validation_batch':8}}
  write(stage/'provenance.json',prov)
  parent.train(m,ls,criterion,opt,device,stage,c,prov,g,resume=False)
  assert training_ready(stage)
  print('TRAIN_COMPLETE',name,flush=True)
 finally:close(ls)
def controller():
 OUT.mkdir(parents=True,exist_ok=True)
 with (OUT/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  deadline=time.time()+30*86400
  for f,v in ARMS:
   for phase in ('preflight','train','test'):
    assert time.time()<deadline
    name=f+'__'+v; work=OUT/name;work.mkdir(exist_ok=True)
    status={'arm':name,'phase':phase,'controller_pid':os.getpid(),'time':time.time(),'deadline':deadline}
    write(OUT/'queue_status.json',status)
    with (work/(phase+'.log')).open('x') as log:
     p=subprocess.Popen([sys.executable,'-B','-u',__file__,'phase',f,v,phase],stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
     status['child_pid']=p.pid;write(OUT/'queue_status.json',status);rc=p.wait()
    write(work/(phase+'_exit.json'),{'exitcode':rc,'pid':p.pid,'time':time.time()})
    if rc:write(OUT/'queue_status.json',dict(status,status='failed',exitcode=rc));return
  write(OUT/'queue_status.json',{'status':'complete','arms':len(ARMS),'time':time.time()})
if __name__=='__main__':
 if sys.argv[1]=='controller':controller()
 else:run(*sys.argv[2:])

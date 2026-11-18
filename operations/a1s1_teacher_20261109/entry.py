"""GPU0: train ISIC RMT-T U-Net, then two students on ISIC followed by HAM."""
import copy, csv, fcntl, hashlib, json, os, shutil, subprocess, sys, time, traceback, types
from pathlib import Path

ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
RUN=Path('/media/8T/PuMengYu/experiments/skin_segmentation/A1S1_TEACHER_HAM_ISIC_20261109')
PY='/home/PuMengYu/anaconda3/envs/medseg/bin/python'
GPU='GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde'
PARENTS={'S1':Path('/media/8T/PuMengYu/experiments/skin_segmentation/A1_S1_HAM_ISIC_FRESH300_20261108/protocol.json'), 'SKIP':Path('/media/8T/PuMengYu/experiments/skin_segmentation/A1_S1_Skip1_v06_HAM_ISIC_FRESH300_20261109_r2/protocol.json')}
HAM_TEACHER=ROOT/'log/八模型对比/runs/protocol_v5_rmt_extension_seed42/rmt_t/seed42/checkpoints/best.pth'
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
from operations.a1_teacher_kd_v1_r2_20261109 import entry as kd
from operations.a1_s1_20261108 import entry as common

def write(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def sha(p):return kd.sha(p)
def read(p):return json.loads(Path(p).read_text())
def imports():
 from pretrained_unet2d import prepare_imports
 prepare_imports('mambavision_t');prepare_imports('tinyvim_s')
 import torch,ham_fast_runner as runner
 return torch,runner
def student_build(arm,pretrained):
 if arm=='S1':from source.a1_s1_20261108.candidate import build_candidate
 else:from source.a1_s1_skip1_v06_20261108.candidate import build_candidate
 return build_candidate(pretrained=pretrained)
def teacher_build(pretrained):
 # Register the unchanged RMT numerical module with the same AST-only unused-import adaptation as the existing RMT runs.
 from operations.encoder_swap_A016_20261107.encoder_swap_candidate import build_candidate
 from rmt_unet2d import RMTUNet2D
 if 'ham_official_rmt' not in sys.modules:
  tmp=build_candidate('rmt_t__A1',False);del tmp
 return RMTUNet2D('rmt_t',pretrained=pretrained)
def teacher_for(ds,torch):
 if ds=='HAM10000':return kd.make_teacher(torch)
 model=teacher_build(False);cp=RUN/'TEACHER_ISIC'/'run/checkpoints/best.pth'
 model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False),strict=True)
 for q in model.parameters():q.requires_grad_(False)
 return model.eval()
def prepare():
 assert not RUN.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')==''
 torch,runner=imports();torch.set_num_threads(1);api=runner.baseline_api();parents={k:read(v) for k,v in PARENTS.items()}
 for parent in parents.values():
  for f,h in parent['protected_sha256'].items():assert sha(f)==h, 'source drift '+f
 for ds in ['ISIC2018','HAM10000']:
  a=parents['S1']['datasets'][ds];b=parents['SKIP']['datasets'][ds]
  assert a['split_sha256']==b['split_sha256'] and sha(a['config']['data']['split_file'])==a['split_sha256']
 from source.isic2018 import data
 assert data.verify()['counts']=={'train':1780,'val':258,'test':518}
 records={}
 for arm in ['S1','SKIP']:
  api.seed_all(42);m=student_build(arm,True);d=kd.digest(m.state_dict());assert d==parents[arm]['model']['ISIC2018']['initial_state_sha256'];records[arm]={'initial_state_sha256':d,'parameters':sum(x.numel() for x in m.parameters())};del m
 api.seed_all(42);t=teacher_build(True);records['TEACHER_ISIC']={'initial_state_sha256':kd.digest(t.state_dict()),'parameters':sum(x.numel() for x in t.parameters()),'encoder_loading_audit':t.encoder.loading_audit};del t
 t=teacher_for('HAM10000',torch);assert all(not q.requires_grad for q in t.parameters()) and not torch.cuda.is_initialized();del t
 assert shutil.disk_usage('/media/8T').free>15*2**30
 sources={str(Path(__file__)):sha(__file__),str(HAM_TEACHER):sha(HAM_TEACHER)}
 for parent in parents.values():sources.update(parent['protected_sha256'])
 for f in [Path(kd.__file__),Path(common.__file__),ROOT/'source/comparison/rmt_unet2d.py',ROOT/'references/code/RMT/classfication_release/RMT.py',ROOT/'operations/encoder_swap_A016_20261107/encoder_swap_candidate.py']:
  sources[str(f)]=sha(f)
 for mod in list(sys.modules.values()):
  f=getattr(mod,'__file__',None)
  if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py') and Path(f).is_file():sources[str(Path(f).resolve())]=sha(f)
 for v in records['TEACHER_ISIC']['encoder_loading_audit'].values():
  if isinstance(v,str) and Path(v).is_file():sources[v]=sha(v)
 RUN.mkdir();protocol={'parents':{k:str(v) for k,v in PARENTS.items()},'datasets':parents['S1']['datasets'],'models':records,'protected_sha256':sources,'teacher_ham':str(HAM_TEACHER),'teacher_isic':'TEACHER_ISIC/run/checkpoints/best.pth','kd':'output soft BCE; epoch 1-50 0.1; 51-249 0.1*(250-epoch)/200; >=250 zero','schedule':['TEACHER_ISIC','S1_ISIC2018','SKIP_ISIC2018','S1_HAM10000','SKIP_HAM10000'],'gpu':GPU,'seed':42,'evaluation':'validation minimum-loss best; fixed test noTTA and mirror4; artifact audit'}
 write(RUN/'protocol.json',protocol);write(RUN/'prepare_complete.json',{'passed':True,'cuda_initialized':False,'teacher_strict_ham':True,'protocol_sha256':sha(RUN/'protocol.json')})
 print('CPU_PREPARE_COMPLETE',flush=True)
def verify():
 p=read(RUN/'protocol.json')
 for f,h in p['protected_sha256'].items():assert sha(f)==h,'source drift '+f
 for ds in ['ISIC2018','HAM10000']:
  c=p['datasets'][ds];assert sha(c['config']['data']['split_file'])==c['split_sha256']
 assert shutil.disk_usage(RUN).free>8*2**30
 return p
def phase(name,kind):
 p=verify();torch,runner=imports();torch.set_num_threads(4);api=runner.baseline_api();api.seed_all(42)
 is_teacher=name=='TEACHER_ISIC';arm='TEACHER_ISIC' if is_teacher else name.split('_')[0];ds='ISIC2018' if is_teacher else name.split('_')[1]
 cfg=p['datasets'][ds]['config'];work=RUN/name;stage=work/('preflight' if kind=='preflight' else 'run')
 model=teacher_build(kind!='test') if is_teacher else student_build(arm,kind!='test')
 if kind!='test':assert kd.digest(model.state_dict())==p['models'][arm]['initial_state_sha256']
 if kind=='test':return test(name,ds,p,model,torch,api)
 if kind=='train':assert read(work/'preflight/acceptance.json')['status']=='passed'
 stage.mkdir(parents=True,exist_ok=False);torch.cuda.set_device(0);device=torch.device('cuda:0');model.cuda()
 loaders,generator=common.data_loaders(ds,cfg,api);opt=torch.optim.AdamW(model.parameters(),lr=cfg['training']['lr'],weight_decay=cfg['training']['weight_decay'],betas=tuple(cfg['training']['betas']),eps=cfg['training']['eps']);criterion=api.BceDiceLoss().cuda()
 info={'model':name,'dataset':ds,'seed':42,'pid':os.getpid(),'physical_gpu':0,'gpu_uuid':GPU,'split_sha256':p['datasets'][ds]['split_sha256'],'protocol_sha256':sha(RUN/'protocol.json'),'test_data_accessed':False,'fresh_start':True}
 write(stage/'provenance.json',info);write(stage/'weight_loading_audit.json',model.encoder.loading_audit)
 handles=[];stats={};epoch={'n':0}
 if not is_teacher:
  teacher=teacher_for(ds,torch).cuda().eval();cp=HAM_TEACHER if ds=='HAM10000' else RUN/'TEACHER_ISIC/run/checkpoints/best.pth';info['teacher_checkpoint_sha256']=sha(cp);write(stage/'provenance.json',info)
  criterion,stats,epoch,hs,ht=kd.attach_kd(model,teacher,torch,device,stage,criterion);handles=[hs,ht]
 try:
  if kind=='preflight':
   proof=runner.preflight(model,loaders,criterion,opt,device,stage,[])
   if not is_teacher:assert stats[1][3]==3 and stats[1][1]>0 and not teacher.training and all(q.grad is None for q in teacher.parameters());proof.update(teacher_frozen=True,kd_preflight=True)
   proof.update(dataset=ds,physical_gpu=0,gpu_uuid=GPU,test_data_accessed=False);write(stage/'acceptance.json',proof);print('PREFLIGHT_PASS',name,flush=True)
  else:
   if not is_teacher:
    original=model.train;last=[0]
    def persist(self,mode=True):
     if mode and epoch['n']>0:kd.write_kd_stats(stats,stage,last[0]);last[0]=epoch['n']
     return original(mode)
    model.train=types.MethodType(persist,model)
   runner.train(model,loaders,criterion,opt,device,stage,cfg,info,generator,resume=False)
   if not is_teacher:kd.write_kd_stats(stats,stage,last[0]);assert len(list(csv.DictReader((stage/'kd_loss_by_epoch.csv').open())))==300
   audit=read(stage/'artifact_audit.json');assert audit['training_complete'] and audit['validation_artifacts_complete'] and audit['epochs']==300
 finally:
  for h in handles:h.remove()
  for loader in loaders.values():
   if hasattr(loader,'close'):loader.close()
   elif getattr(loader,'_iterator',None) is not None:loader._iterator._shutdown_workers()
def test(name,ds,p,model,torch,api):
 from torch.utils.data import DataLoader
 from final_test_pipeline import fixed_best_record,validate_outputs
 from operations.nnunet_mirror_validation.run_validation import Predictor
 work=RUN/name;fixed=fixed_best_record(work/'run');assert fixed['history_rows']==300
 out=work/'final_test';out.mkdir(exist_ok=False);cp=out/'fixed_best.pth';shutil.copy2(fixed['checkpoint'],cp);assert sha(cp)==fixed['checkpoint_sha256']
 model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False),strict=True)
 write(out/'cpu_load_audit.json',{'strict':True,'checkpoint_sha256':sha(cp),'selected_epoch':fixed['selected_epoch'],'cuda_initialized':torch.cuda.is_initialized()})
 if ds=='HAM10000':
  from project_paths import PATHS
  dataset=api.HAMDataset(Path(PATHS['active_split']).parent,'test');expected=2020
 else:
  from source.isic2018 import data
  dataset=data.ISICDataset(data.CACHE,'test');expected=518
 ids={r['image_id'] for r in dataset.rows};assert len(dataset)==len(ids)==expected
 loader=DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0);torch.cuda.set_device(0);model.cuda().eval();results={}
 for mode in ['none','mirror4']:
  target=out/mode;pred=Predictor(model,mode=='mirror4').eval();result=api.evaluate(pred,loader,api.BceDiceLoss().cuda(),torch.device('cuda:0'),target)
  assert len(pred.times)==expected and pred.verify_remaining==0
  result.update(dataset=ds,model=name,split='test',seed=42,selected_epoch=fixed['selected_epoch'],checkpoint=str(cp),checkpoint_sha256=sha(cp),split_sha256=p['datasets'][ds]['split_sha256'],tta=mode=='mirror4',training_epochs=300,test_used_for_checkpoint_selection=False,physical_gpu=0,gpu_uuid=GPU)
  audit=validate_outputs(target,ids,result);assert audit['complete'];write(target/'summary.json',result);write(target/'artifact_audit.json',audit);pred.close();results[mode]=result
 write(work/'completion.json',{'complete':True,'epochs':300,'selected_epoch':fixed['selected_epoch'],'test_images_per_mode':expected,'results':results})
def queue():
 with (RUN/'queue.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for name in verify()['schedule']:
   work=RUN/name;work.mkdir(exist_ok=True)
   for kind in ['preflight','train','test']:
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=GPU,EXPERIMENT_PHYSICAL_GPU='0',EXPERIMENT_GPU_UUID=GPU,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
    with (work/(kind+'.log')).open('x') as log:
     child=subprocess.Popen([PY,'-B','-u',__file__,'phase',name,kind],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
     write(RUN/'queue_status.json',{'status':kind,'experiment':name,'pid':child.pid,'controller_pid':os.getpid(),'gpu':0,'time':time.time()});rc=child.wait()
    write(work/(kind+'_exitcode.json'),{'exitcode':rc,'pid':child.pid});assert rc==0,(name,kind,rc)
  write(RUN/'queue_status.json',{'status':'complete','time':time.time()})
if __name__=='__main__':
 try:
  if sys.argv[1]=='prepare':prepare()
  elif sys.argv[1]=='phase':phase(sys.argv[2],sys.argv[3])
  elif sys.argv[1]=='queue':queue()
 except BaseException as e:
  if RUN.exists():write(RUN/'failure.json',{'error':repr(e),'traceback':traceback.format_exc(),'time':time.time()})
  raise

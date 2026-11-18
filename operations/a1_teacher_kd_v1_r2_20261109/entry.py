"""A1-teacher: frozen RMT-T output KD on HAM10000; fresh A1 student."""
import ast, copy, csv, fcntl, hashlib, importlib.util, json, os, shutil, subprocess, sys, time, traceback, types
from pathlib import Path

ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
OPS=Path(__file__).resolve().parent
RUN=Path('/media/8T/PuMengYu/experiments/skin_segmentation/A1_Teacher_RMT-T_KD-V1_HAM_FRESH300_20261109_r2')
PY='/home/PuMengYu/anaconda3/envs/medseg/bin/python'
UUIDS=['GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde','GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b']
REF=Path('/media/8T/PuMengYu/experiments/skin_segmentation/DecoderDualA_RESUME100to300_20261106/protocol.json')
TEACHER=Path('/home/PuMengYu/experiments/skin_segmentation/log/八模型对比/runs/protocol_v5_rmt_extension_seed42/rmt_t/seed42/checkpoints/best.pth')
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]

def read(p): return json.loads(Path(p).read_text())
def write(p,o):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def digest(sd):
 h=hashlib.sha256()
 for n,t in sorted(sd.items()):h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def imports():
 from pretrained_unet2d import prepare_imports
 prepare_imports('mambavision_t');prepare_imports('tinyvim_s')
 import torch, ham_fast_runner as runner
 from ham_fast_pipeline import make_fast_loaders
 from source.decoder_dual_a_20261106.candidate import build_candidate
 return torch,runner,make_fast_loaders,build_candidate
def make_teacher(torch):
 # RMT's upstream module has one unused fvcore FLOP-report import absent here.
 from rmt_unet2d import SOURCE
 if 'ham_official_rmt' not in sys.modules:
  tree=ast.parse(SOURCE.read_text());names={'FlopCountAnalysis','flop_count_table'}
  assert not any(isinstance(n,ast.Name) and n.id in names for n in ast.walk(tree))
  imports=[n for n in tree.body if isinstance(n,ast.ImportFrom) and n.module=='fvcore.nn']
  assert len(imports)==1 and {a.name for a in imports[0].names}==names
  tree.body.remove(imports[0]);mod=types.ModuleType('ham_official_rmt');mod.__file__=str(SOURCE);sys.modules[mod.__name__]=mod
  exec(compile(tree,str(SOURCE),'exec'),mod.__dict__)
 from rmt_unet2d import RMTUNet2D
 teacher=RMTUNet2D('rmt_t',pretrained=False)
 state=torch.load(TEACHER,map_location='cpu',weights_only=False)
 teacher.load_state_dict(state,strict=True)
 for p in teacher.parameters():p.requires_grad_(False)
 teacher.eval()
 return teacher
def verify():
 p=read(RUN/'protocol.json')
 for f,h in p['protected_sha256'].items():assert sha(f)==h,'source drift: '+f
 assert sha(p['config']['data']['split_file'])==p['config']['data']['split_sha256']
 assert sha(TEACHER)==p['teacher']['checkpoint_sha256']
 assert shutil.disk_usage(RUN).free>8*2**30
 return p
def prepare():
 assert not RUN.exists(),'refuse to overwrite existing run'
 assert REF.is_file() and TEACHER.is_file()
 torch,runner,_,build=imports();torch.set_num_threads(1)
 parent=read(REF)
 for rel,h in parent['source_sha256'].items():assert sha(ROOT/rel)==h,'parent source drift '+rel
 cfg=copy.deepcopy(parent['config']);assert cfg['data']['counts']=={'train':6983,'val':1010,'test':2020}
 assert cfg['training']['epochs']==300 and cfg['training']['batch_size']==8
 api=runner.baseline_api();api.seed_all(42);student=build(pretrained=True)
 initial=digest(student.state_dict());params=sum(p.numel() for p in student.parameters())
 assert all(p.requires_grad for p in student.parameters())
 sd=student.state_dict();buf=__import__('io').BytesIO();torch.save(sd,buf);buf.seek(0);student.load_state_dict(torch.load(buf,map_location='cpu',weights_only=False),strict=True)
 teacher=make_teacher(torch);teacher_hash=sha(TEACHER);teacher_params=sum(p.numel() for p in teacher.parameters())
 assert not any(p.requires_grad for p in teacher.parameters()) and not torch.cuda.is_initialized()
 del teacher,student,sd,buf
 RUN.mkdir(parents=True);(RUN/'HAM10000').mkdir()
 protected={str(REF):sha(REF),str(TEACHER):teacher_hash,str(Path(__file__)):sha(__file__)}
 for mod in list(sys.modules.values()):
  f=getattr(mod,'__file__',None)
  if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py') and Path(f).is_file():protected[str(Path(f).resolve())]=sha(f)
 protocol=dict(identity='A1-teacher｜HAM10000｜A1 fresh300 + frozen RMT-T output KD-V1',name='A1-teacher',dataset='HAM10000',config=cfg,
  student=dict(model='A1',architecture='base A1 from source.decoder_dual_a_20261106.candidate.build_candidate',parameters=params,initial_state_sha256=initial,initialization='seed42; official ImageNet TinyViM encoder weights; no HAM segmentation checkpoint; same fresh initialization rule as A1 control'),
  teacher=dict(model='RMT-T + U-Net',parameters=teacher_params,checkpoint=str(TEACHER),checkpoint_sha256=teacher_hash,historical_test_global_dice=94.9793,frozen=True,eval_mode=True,strict_full_state_load=True),
  distillation=dict(kind='final output pixelwise soft-target BCE',temperature=1,feature_distillation=False,teacher_grad=False,student_head='capture pre-sigmoid head logits; bilinear-upsample to the student prediction size',teacher_head='capture pre-sigmoid head logits; apply sigmoid once to form detached soft target',lambda_schedule='epoch 1-50: 0.1; 51-249: 0.1*(250-epoch)/200; epoch >=250: 0',final_50_epochs_kd_off=True),
  training='HAM parent protocol unchanged: seed42, fresh300, batch8, FP32, AdamW and cosine schedule, original augmentation/sampling/validation and BCE+Dice unchanged; all A1 student parameters trainable; select best by validation loss; no test data in selection',
  evaluation='fixed validation-best checkpoint; strict CPU load; HAM test2020 no-TTA and mirror4; student only; complete prediction and artifact audits',
  protected_sha256=protected,scheduling=dict(shared_gpu_authorized=True,allowed_physical_gpus=[0,1],preflight_gate='at most 2 existing compute processes and >=14000 MiB free; shared admission lock; do not preempt existing jobs',poll_seconds=30),created_unix=time.time())
 write(RUN/'protocol.json',protocol)
 write(RUN/'cpu_acceptance.json',dict(passed=True,cuda_initialized=False,student_initial_state_sha256=initial,student_parameters=params,teacher_checkpoint_sha256=teacher_hash,teacher_strict_load=True,teacher_parameters=teacher_params,teacher_frozen=True,split_counts=cfg['data']['counts'],test_data_accessed=False))
 for f,h in protected.items():
  dst=RUN/'source_snapshot'/Path(f).relative_to('/');dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dst);assert sha(dst)==h
 write(RUN/'prepare_complete.json',dict(passed=True,protocol_sha256=sha(RUN/'protocol.json'),source_count=len(protected),cuda_initialized=False))
 print('A1_TEACHER_CPU_PREPARE_COMPLETE',params,teacher_params,initial,flush=True)
def lambda_kd(epoch):
 if epoch<=50:return .1
 if epoch<250:return .1*(250-epoch)/200
 return 0.
def attach_kd(student,teacher,torch,device,run_dir,base_criterion):
 import torch.nn.functional as F
 from types import MethodType
 cache={};stats={};epoch={'n':0};orig_forward=student.forward;orig_train=student.train
 def student_hook(module,args,out):cache['zs']=out
 def teacher_hook(module,args,out):cache['zt']=out
 hs=student.decoder.head.register_forward_hook(student_hook);ht=teacher.decoder.head.register_forward_hook(teacher_hook)
 def wrapped_forward(self,x):
  if self.training:
   teacher.eval()
   with torch.no_grad():teacher(x)
  pred=orig_forward(x)
  if not self.training:return pred
  if 'zs' not in cache or 'zt' not in cache:raise RuntimeError('missing pre-sigmoid head outputs')
  cache['zs']=F.interpolate(cache['zs'],size=pred.shape[-2:],mode='bilinear',align_corners=False)
  cache['pt']=torch.sigmoid(cache['zt']).detach()
  assert cache['zs'].shape==cache['pt'].shape==pred.shape
  return pred
 def wrapped_train(self,mode=True):
  if mode:epoch['n']+=1
  return orig_train(mode)
 student.forward=MethodType(wrapped_forward,student);student.train=MethodType(wrapped_train,student)
 class Criterion:
  def __call__(self,pred,target):
   seg=base_criterion(pred,target)
   if not student.training:return seg
   e=epoch['n'];lam=lambda_kd(e)
   kd=F.binary_cross_entropy_with_logits(cache['zs'],cache['pt'])
   if not torch.isfinite(kd):raise FloatingPointError('nonfinite KD loss')
   rec=stats.setdefault(e,[0.,0.,0.,0]);rec[0]+=float(seg.detach());rec[1]+=float(kd.detach());rec[2]+=lam*float(kd.detach());rec[3]+=1
   return seg+lam*kd
 wrapped_criterion=Criterion()
 wrapped_criterion.wb=base_criterion.wb
 wrapped_criterion.wd=base_criterion.wd
 return wrapped_criterion,stats,epoch,hs,ht
def write_kd_stats(stats,run_dir,last_epoch=0):
 path=run_dir/'kd_loss_by_epoch.csv';exists=path.exists()
 with path.open('a',newline='') as f:
  w=csv.writer(f)
  if not exists:w.writerow(['epoch','L_seg_mean','L_KD_mean','lambda','lambda_x_L_KD_mean','batches'])
  for e in sorted(k for k in stats if k>last_epoch):
   seg,kd,weighted,n=stats[e];w.writerow([e,seg/n,kd/n,lambda_kd(e),weighted/n,n])
def phase(kind,gpu):
 p=verify();torch,runner,loaders_fn,build=imports();torch.set_num_threads(4);api=runner.baseline_api();api.seed_all(42)
 student=build(pretrained=True);assert digest(student.state_dict())==p['student']['initial_state_sha256']
 if kind=='test':return test(gpu,p,torch,runner,api,student)
 teacher=make_teacher(torch)
 work=RUN/'HAM10000';run_dir=work/'run';torch.cuda.set_device(0);device=torch.device('cuda:0')
 from ham_fast_pipeline import make_fast_loaders
 loaders,generator=loaders_fn(p['config'],api,workers=2,validation_batch=8)
 student.cuda();teacher.cuda().eval();criterion_seg=api.BceDiceLoss().cuda();opt=torch.optim.AdamW(student.parameters(),lr=p['config']['training']['lr'],weight_decay=p['config']['training']['weight_decay'],betas=tuple(p['config']['training']['betas']),eps=p['config']['training']['eps'])
 info=dict(model='A1-teacher',dataset='HAM10000',seed=42,physical_gpu=gpu,gpu_uuid=UUIDS[gpu],pid=os.getpid(),protocol_sha256=sha(RUN/'protocol.json'),split_sha256=p['config']['data']['split_sha256'],teacher_checkpoint_sha256=p['teacher']['checkpoint_sha256'],student_initial_state_sha256=p['student']['initial_state_sha256'],test_data_accessed=False,training=p['config']['training'])
 if kind=='preflight':
  (work/'preflight').mkdir(parents=True,exist_ok=True)
  kdcrit,kdstats,_,hs,ht=attach_kd(student,teacher,torch,device,work/'preflight',criterion_seg)
  proof=runner.preflight(student,loaders,kdcrit,opt,device,work/'preflight',[])
  assert kdstats[1][3]==3 and kdstats[1][1]>0
  assert all(q.grad is None for q in teacher.parameters()) and not teacher.training
  proof.update(kd_preflight=True,kd_preflight_steps=kdstats[1][3],kd_loss_mean=kdstats[1][1]/kdstats[1][3],teacher_output='logits captured before sigmoid; sigmoid once for detached target',student_output='logits captured before sigmoid; bilinear resize to 256x256',teacher_frozen=all(not q.requires_grad for q in teacher.parameters()),teacher_eval=not teacher.training,gpu_uuid=UUIDS[gpu],physical_gpu=gpu,test_data_accessed=False)
  hs.remove();ht.remove()
  write(work/'preflight/acceptance.json',proof);print('A1_TEACHER_KD_PREFLIGHT',json.dumps(proof),flush=True)
 else:
  assert read(work/'preflight/acceptance.json')['kd_preflight']
  run_dir.mkdir(parents=True,exist_ok=False);write(run_dir/'provenance.json',info)
  kdcrit,stats,epoch,hs,ht=attach_kd(student,teacher,torch,device,run_dir,criterion_seg)
  orig_train=student.train;last=[0]
  def persist_train(self,mode=True):
   if mode and epoch['n']>0:
    write_kd_stats(stats,run_dir,last[0]);last[0]=epoch['n']
   return orig_train(mode)
  student.train=types.MethodType(persist_train,student)
  try:
   runner.train(student,loaders,kdcrit,opt,device,run_dir,p['config'],info,generator,resume=False)
   write_kd_stats(stats,run_dir,last[0])
   audit=read(run_dir/'artifact_audit.json');assert audit['training_complete'] and audit['epochs']==300
   assert len(list(csv.DictReader((run_dir/'kd_loss_by_epoch.csv').open())))==300
  finally:hs.remove();ht.remove()
 for ld in loaders.values():
  if hasattr(ld,'close'):ld.close()
def test(gpu,p,torch,runner,api,student):
 from torch.utils.data import DataLoader
 from final_test_pipeline import fixed_best_record,validate_outputs
 from operations.nnunet_mirror_validation.run_validation import Predictor
 work=RUN/'HAM10000';fixed=fixed_best_record(work/'run');assert fixed['history_rows']==300
 out=work/'final_test';out.mkdir(exist_ok=False);cp=out/'fixed_best.pth';shutil.copy2(fixed['checkpoint'],cp);assert sha(cp)==fixed['checkpoint_sha256']
 student.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False),strict=True)
 write(out/'cpu_load_audit.json',dict(strict=True,checkpoint_sha256=sha(cp),selected_epoch=fixed['selected_epoch'],cuda_initialized=torch.cuda.is_initialized()))
 from project_paths import PATHS
 ds=api.HAMDataset(Path(PATHS['active_split']).parent,'test');ids={r['image_id'] for r in ds.rows};assert len(ds)==len(ids)==2020
 loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=0);student.cuda().eval();results={}
 for mode in ['none','mirror4']:
  target=out/mode;target.mkdir();pred=Predictor(student,mode=='mirror4').eval();res=api.evaluate(pred,loader,api.BceDiceLoss().cuda(),torch.device('cuda:0'),target)
  assert len(pred.times)==2020 and pred.verify_remaining==0
  res.update(dataset='HAM10000',model='A1-teacher',split='test',seed=42,selected_epoch=fixed['selected_epoch'],checkpoint=str(cp),checkpoint_sha256=sha(cp),split_sha256=p['config']['data']['split_sha256'],tta=mode=='mirror4',training_epochs=300,test_used_for_checkpoint_selection=False,teacher_used_at_inference=False,physical_gpu=gpu,gpu_uuid=UUIDS[gpu])
  audit=validate_outputs(target,ids,res);assert audit['complete'];write(target/'summary.json',res);write(target/'artifact_audit.json',audit);pred.close();results[mode]=res
 write(work/'completion.json',dict(complete=True,epochs=300,selected_epoch=fixed['selected_epoch'],results=results,test_images=2020,teacher_used_at_inference=False))
def free_mib(gpu):
 s=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader'],text=True)
 return int(next(x.split(',')[1].strip().split()[0] for x in s.splitlines() if x.startswith(UUIDS[gpu])))
def apps(gpu):
 s=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 return [int(x.split(',')[1]) for x in s.splitlines() if x.startswith(UUIDS[gpu])]
def controller():
 gate_path=ROOT/'operations/encoder_swap_3slot_takeover_20261108/admission.lock'
 for kind in ['preflight','train','test']:
  while True:
   with gate_path.open('a') as gate:
    fcntl.flock(gate,fcntl.LOCK_EX)
    chosen=next((g for g in [0,1] if len(apps(g))<=2 and free_mib(g)>=14000),None)
    if chosen is not None:
     env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=UUIDS[chosen],EXPERIMENT_PHYSICAL_GPU=str(chosen),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
     with (RUN/'HAM10000'/f'{kind}.log').open('a') as log:
      child=subprocess.Popen([PY,'-B','-u',__file__,'phase',kind,str(chosen)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
      write(RUN/'HAM10000'/'queue_status.json',dict(status=kind,physical_gpu=chosen,pid=child.pid,updated_unix=time.time()))
      deadline=time.time()+180
      while child.poll() is None and child.pid not in apps(chosen) and time.time()<deadline:time.sleep(1)
      if child.poll() is not None or child.pid not in apps(chosen):raise RuntimeError(f'{kind} did not attach to assigned GPU')
     fcntl.flock(gate,fcntl.LOCK_UN)
    else:
     fcntl.flock(gate,fcntl.LOCK_UN);write(RUN/'HAM10000'/'queue_status.json',dict(status='waiting_resources',phase=kind,compute_pids={str(g):apps(g) for g in (0,1)},free_mib={str(g):free_mib(g) for g in (0,1)},updated_unix=time.time()))
   if chosen is not None:
    rc=child.wait();write(RUN/'HAM10000'/f'{kind}_exitcode.json',dict(exitcode=rc,pid=child.pid,gpu=chosen,finished_unix=time.time()));assert rc==0,kind+' failed; later phases blocked';break
   time.sleep(30)
 write(RUN/'HAM10000'/'queue_status.json',dict(status='complete',updated_unix=time.time()))
if __name__=='__main__':
 try:
  if sys.argv[1]=='prepare':prepare()
  elif sys.argv[1]=='queue':controller()
  elif sys.argv[1]=='phase':phase(sys.argv[2],int(sys.argv[3]))
 except BaseException as e:
  if RUN.exists():write(RUN/'HAM10000'/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),pid=os.getpid(),time=time.time()))
  raise

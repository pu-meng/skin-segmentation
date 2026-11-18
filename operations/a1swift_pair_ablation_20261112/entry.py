import copy,fcntl,hashlib,json,os,subprocess,sys,time,importlib
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
OPS=ROOT/'operations/a1swift_pair_ablation_20261112'
RUN=Path('/media/8T/PuMengYu/experiments/skin_segmentation/A1Swift_PairAblations_ISIC_FRESH300_20261112')
PY='/home/PuMengYu/anaconda3/envs/medseg/bin/python'
UUIDS=['GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde','GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b']
BASE=Path('/media/8T/PuMengYu/experiments/skin_segmentation/ISIC2018_A0_A1_FRESH300_20261108/protocol.json')
ARMS={'no_e2_no_d2d3':0,'no_skip1_no_e2':1}
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba'),str(ROOT/'source/isic2018'),str(ROOT/'operations/encoder_swap_A016_20261107')]
from operations.isic_a01_20261108 import entry as common
common.RUN=RUN;common.UUIDS=UUIDS

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str)+'\n');t.replace(p)

def build(arm,pretrained=True):
 import torch
 from torch import nn
 from torch.nn import functional as F
 from operations.encoder_swap_A016_20261107.encoder_swap_candidate import build_candidate as swap
 from Models.TinyViM.four_stage_decoder import build_four_stage_baseline
 model=swap('swiftformer_s__A1',pretrained=pretrained);old=model.decoder
 remove_e2=arm in ('no_e2_no_d2d3','no_skip1_no_e2')
 remove_e3=arm=='no_skip1_no_e2';remove_d=arm=='no_e2_no_d2d3'
 stages=old.stages
 if remove_d:
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(4204);plain=build_four_stage_baseline(pretrained=False)
   stages[2]=plain.decoder.stages[2];stages[3]=plain.decoder.stages[3]
 class PairDecoder(nn.Module):
  def __init__(self):
   super().__init__();self.stages=stages;self.ups=old.ups;self.head=old.head
   self.e2=None if remove_e2 else old.e2;self.e3=None if remove_e3 else old.e3
  def forward(self,f):
   x=self.stages[3](f[3] if self.e2 is None else self.e2(f[3]))
   for level in (2,1,0):
    x=self.ups[level](x);s=f[level]
    if level==0 and self.e3 is not None:s,x=self.e3(s,x)
    x=self.stages[level](torch.cat([x,s],1))
   return torch.sigmoid(F.interpolate(self.head(x),size=(256,256),mode='bilinear',align_corners=False))
 model.decoder=PairDecoder()
 d=model.decoder
 assert (d.e2 is None) and ((d.e3 is None)==remove_e3)
 if remove_d:
  from Models.TinyViM.model import DoubleConv
  assert type(d.stages[2]) is DoubleConv and type(d.stages[3]) is DoubleConv
 else:
  assert type(d.stages[2]).__name__=='DualStage' and type(d.stages[3]).__name__=='DStage'
 assert all(p.requires_grad for p in model.parameters())
 return model

def imports(arm):
 import torch,ham_fast_runner as runner
 from source.isic2018 import data
 from pretrained_unet2d import prepare_imports
 prepare_imports('mambavision_t');prepare_imports('tinyvim_s')
 return torch,runner,data,lambda pretrained:build(arm,pretrained)
common.imports=imports

def verify():
 p=json.loads((RUN/'protocol.json').read_text())
 for f,h in p['protected_sha256'].items():assert sha(f)==h,'source drift: '+f
 assert sha(p['config']['data']['split_file'])==p['config']['data']['split_sha256']
 return p
common.verify=verify

def prepare():
 import torch,ham_fast_runner as runner
 from source.isic2018 import data
 assert not RUN.exists(),'refuse existing run root'
 parent=json.loads(BASE.read_text());cfg=copy.deepcopy(parent['config']);audit=json.loads((data.DEDUP/'split_audit.json').read_text())
 assert data.verify()['counts']==cfg['data']['counts']==dict(train=1780,val=258,test=518)
 assert audit['checks']['unresolved_review_pairs']==audit['checks']['remaining_multi_active_original_strong_components']==0
 assert sha(data.DEDUP/'split.csv')==audit['split_sha256']==cfg['data']['dedup_split_sha256']
 assert cfg['training']['seed']==42 and cfg['training']['epochs']==300 and cfg['training']['total_optimizer_updates']==66900
 api=runner.baseline_api();records={};RUN.mkdir(parents=True)
 for arm in ARMS:
  api.seed_all(42);torch.set_num_threads(1);m=build(arm,True);d=m.decoder
  buf=__import__('io').BytesIO();torch.save(m.state_dict(),buf);buf.seek(0);m.load_state_dict(torch.load(buf,map_location='cpu',weights_only=False),strict=True)
  assert not torch.cuda.is_initialized()
  records[arm]=dict(initial_state_sha256=common.digest(m.state_dict()),parameters=sum(p.numel() for p in m.parameters()),state_bytes=sum(t.numel()*t.element_size() for t in m.state_dict().values()),e2_absent=d.e2 is None,e3_absent=d.e3 is None,d2=type(d.stages[2]).__name__,d3=type(d.stages[3]).__name__,strict_roundtrip=True,all_trainable=all(p.requires_grad for p in m.parameters()),assignment_gpu=ARMS[arm])
  assert records[arm]['all_trainable'] and not torch.cuda.is_initialized()
  (RUN/arm).mkdir();write(RUN/arm/'cpu_acceptance.json',dict(passed=True,**records[arm]));del m,buf
 sources={str(BASE):sha(BASE)}
 for mod in list(sys.modules.values()):
  f=getattr(mod,'__file__',None)
  if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py') and Path(f).is_file():sources[str(Path(f).resolve())]=sha(f)
 for f in [Path(__file__),ROOT/'operations/encoder_swap_A016_20261107/encoder_swap_candidate.py',ROOT/'operations/isic_a01_20261108/entry.py',ROOT/'Models/TinyViM/four_stage_decoder.py',ROOT/'Models/TinyViM/model.py',ROOT/'source/comparison/unet2d.py',ROOT/'source/bottleneck_mamba/final_test_pipeline.py',ROOT/'operations/nnunet_mirror_validation/run_validation.py',ROOT/'operations/nnunet_mirror_validation/mirror_core.py']:sources[str(f)]=sha(f)
 for arm in ARMS:
  audit_load=build(arm,True).encoder.loading_audit
  for v in audit_load.values():
   if isinstance(v,str) and Path(v).is_file():sources[str(Path(v).resolve())]=sha(v)
 needed=sum(r['state_bytes'] for r in records.values())*8+5*2**30
 assert __import__('shutil').disk_usage(RUN).free>needed+8*2**30
 cfg['evaluation']['completion']='300 epochs -> validation-minloss fixed best -> CPU strict load -> ISIC2018 test518 none and mirror4 -> artifact audits'
 protocol=dict(identity='A1+SwiftFormer pairwise interaction ablations on deduplicated ISIC2018; seed42 fresh300',config=cfg,arms=records,data_audit=audit,protected_sha256=sources,contract='Same full A1+SwiftFormer initialization and training recipe; arm1 removes only E2 and D2+D3; arm2 removes only E3 skip and E2; independent fresh training; test never selects checkpoints.',scheduling=dict(gpu0_arm='no_e2_no_d2d3',gpu1_arm='no_skip1_no_e2',admission_lock=str(ROOT/'operations/encoder_swap_3slot_takeover_20261108/admission.lock'),free_vram_gate_mib=8192,max_compute_processes=4))
 write(RUN/'protocol.json',protocol)
 for f,h in sources.items():
  dst=RUN/'source_snapshot'/Path(f).relative_to('/');dst.parent.mkdir(parents=True,exist_ok=True);__import__('shutil').copy2(f,dst);assert sha(dst)==h
 write(RUN/'prepare_complete.json',dict(passed=True,protocol_sha256=sha(RUN/'protocol.json'),arms=records))
 print('CPU_PREPARE_COMPLETE',json.dumps(records,ensure_ascii=False),flush=True)

def phase(arm,mode,gpu):
 common.RUN=RUN;common.UUIDS=UUIDS;common.verify=verify;common.imports=imports
 common.phase(arm,mode,gpu)

def launch(arm,gpu,mode):
 guard=ROOT/'operations/encoder_swap_3slot_takeover_20261108/admission.lock'
 gate=json.loads((RUN/'protocol.json').read_text())['scheduling']['free_vram_gate_mib']
 while True:
  lock=guard.open('a');fcntl.flock(lock,fcntl.LOCK_EX)
  out=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
  pids=[int(x.split(',')[1]) for x in out.splitlines() if x.startswith(UUIDS[gpu])]
  stats=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader'],text=True)
  free=int(next(x.split(',')[1].strip().split()[0] for x in stats.splitlines() if x.startswith(UUIDS[gpu])))
  if len(pids)<4 and free>=gate:break
  lock.close();write(RUN/arm/'controller_status.json',dict(status='waiting_gpu_admission',gpu=gpu,compute_pids=pids,free_mib=free,threshold_mib=gate,updated_unix=time.time()));time.sleep(30)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=UUIDS[gpu],EXPERIMENT_PHYSICAL_GPU=str(gpu),EXPERIMENT_GPU_UUID=UUIDS[gpu],OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 log=(RUN/arm/f'{mode}.log').open('x')
 child=subprocess.Popen([PY,'-B','-u',str(OPS/'entry.py'),'phase',arm,mode,str(gpu)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 write(RUN/arm/'controller_status.json',dict(status=mode,controller_pid=os.getpid(),child_pid=child.pid,physical_gpu=gpu,gpu_uuid=UUIDS[gpu],prior_compute_pids=pids,free_mib_at_admission=free,updated_unix=time.time()))
 fcntl.flock(lock,fcntl.LOCK_UN);lock.close();rc=child.wait();log.close()
 write(RUN/arm/f'{mode}_exit.json',dict(exitcode=rc,pid=child.pid,time=time.time()));assert rc==0,f'{arm}/{mode} failed'

def worker(arm,gpu):
 lock=(RUN/arm/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for mode in ('preflight','train','test'):
  verify();launch(arm,gpu,mode)
 write(RUN/arm/'controller_status.json',dict(status='complete',gpu=gpu,updated_unix=time.time()))

def controller():
 procs=[]
 for arm,gpu in ARMS.items():
  log=(RUN/f'{arm}_controller.log').open('x')
  p=subprocess.Popen([PY,'-B','-u',str(OPS/'entry.py'),'worker',arm,str(gpu)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  procs.append((p,log))
 write(RUN/'queue_status.json',dict(status='workers_started',workers={a:{'gpu':g,'pid':p.pid} for (a,g),(p,_) in zip(ARMS.items(),procs)},updated_unix=time.time()))
 for p,log in procs:
  rc=p.wait();log.close();assert rc==0,'worker failed; see worker log'
 write(RUN/'queue_status.json',dict(status='complete',arms=len(ARMS),updated_unix=time.time()))

if __name__=='__main__':
 if sys.argv[1]=='prepare':prepare()
 elif sys.argv[1]=='controller':controller()
 elif sys.argv[1]=='worker':worker(sys.argv[2],int(sys.argv[3]))
 elif sys.argv[1]=='phase':phase(sys.argv[2],sys.argv[3],int(sys.argv[4]))
 else:raise SystemExit('usage prepare|controller|worker|phase')

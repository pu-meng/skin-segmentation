"""ISIC2018 E0 four-stage decoder with SwiftFormer-S and MobileOne-S1 encoders."""
import copy, hashlib, importlib.util, json, os, shutil, subprocess, sys, time
from pathlib import Path

ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
OPS=Path(__file__).resolve().parent
RUN=Path('/media/8T/PuMengYu/experiments/skin_segmentation/ISIC2018_E0_EncoderSwap_FRESH300_20261108')
REF=ROOT/'log/TinyViM_四层_E0_E1E3_ISIC2018_20261105/protocol.json'
PY='/home/PuMengYu/anaconda3/envs/medseg/bin/python'
UUIDS=['GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde','GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b']
ARMS=('swiftformer_s','mobileone_s1')
WEIGHTS=ROOT/'Models/pretrained/tinyvim_peers_20261101'
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]

def read(p): return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2));t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def digest(sd):
 h=hashlib.sha256()
 for n,t in sorted(sd.items()):h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def build(arm,pretrained=True):
 import torch
 from torch import nn
 from pretrained_unet2d import prepare_imports
 prepare_imports('tinyvim_s')
 from Models.TinyViM.four_stage_decoder import build_four_stage_baseline
 from Models.TinyViMPeers.model import Encoder,load_classifier
 model=build_four_stage_baseline(pretrained=False)
 backbone,audit=load_classifier(arm,pretrained)
 model.encoder=Encoder(arm,backbone);model.encoder.loading_audit=audit
 channels=(96,192,512,1280) if arm=='mobileone_s1' else (48,64,168,224)
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(43)
  model.align=nn.ModuleList(nn.Conv2d(a,b,1) for a,b in zip(channels,(64,128,256,512)))
 assert len(model.decoder.stages)==4 and all(p.requires_grad for p in model.parameters())
 return model

def imports(arm):
 import torch,ham_fast_runner as runner
 from source.isic2018 import data
 return torch,runner,data,lambda pretrained=True:build(arm,pretrained)

def prepare():
 assert not RUN.exists() and os.environ.get('CUDA_VISIBLE_DEVICES','')==''
 from pretrained_unet2d import prepare_imports
 prepare_imports('tinyvim_s')
 import torch,ham_fast_runner as runner
 from source.isic2018 import data
 torch.set_num_threads(1)
 ref=read(REF);meta=data.verify();audit=read(data.DEDUP/'split_audit.json')
 assert meta['counts']==dict(train=1780,val=258,test=518)
 assert sha(data.DEDUP/'split.csv')==audit['split_sha256']==ref['data']['dedup_split_sha256']
 assert sha(data.CACHE/'split.csv')==ref['data']['split_sha256']
 assert audit['checks']['unresolved_review_pairs']==audit['checks']['remaining_multi_active_original_strong_components']==0
 cfg={k:copy.deepcopy(ref[k]) for k in ('data','training','preprocessing','evaluation')}
 assert cfg['training']['epochs']==300 and cfg['training']['total_optimizer_updates']==66900
 RUN.mkdir(parents=True)
 records={}
 api=runner.baseline_api()
 for arm in ARMS:
  api.seed_all(42);m=build(arm,True)
  state=digest(m.state_dict());buf=__import__('io').BytesIO();torch.save(m.state_dict(),buf);buf.seek(0)
  m.load_state_dict(torch.load(buf,map_location='cpu',weights_only=False),strict=True)
  records[arm]=dict(initial_state_sha256=state,parameters=sum(p.numel() for p in m.parameters()),state_bytes=sum(t.numel()*t.element_size() for t in m.state_dict().values()),encoder_loading_audit=m.encoder.loading_audit,model_class='E0 FourStageDecoder + '+arm)
  write(RUN/arm/'cpu_acceptance.json',dict(passed=True,strict_roundtrip=True,cuda_initialized=torch.cuda.is_initialized(),**records[arm]))
  assert not torch.cuda.is_initialized();del m,buf
 sources={str(REF):sha(REF),str(OPS/'run.py'):sha(OPS/'run.py')}
 for f in [ROOT/'Models/TinyViM/four_stage_decoder.py',ROOT/'Models/TinyViMPeers/model.py',ROOT/'source/comparison/pretrained_unet2d.py',ROOT/'source/comparison/ham_fast_runner.py',ROOT/'source/isic2018/data.py',ROOT/'source/bottleneck_mamba/final_test_pipeline.py',ROOT/'operations/nnunet_mirror_validation/run_validation.py',ROOT/'operations/nnunet_mirror_validation/mirror_core.py',ROOT/'operations/isic_a01_20261108/entry.py']:
  if f.exists():sources[str(f)]=sha(f)
 for arm in ARMS:
  audit=records[arm]['encoder_loading_audit']
  path=audit.get('source',{}).get('weights_path') or audit.get('weights_path')
  if path and Path(path).is_file():sources[path]=sha(path)
 needed=sum(r['state_bytes'] for r in records.values())*12+5*2**30
 assert shutil.disk_usage(RUN.parent).free>needed+8*2**30
 p=dict(identity='ISIC2018 E0 four-stage decoder; encoder-only replacement SwiftFormer-S and MobileOne-S1; deduplicated split; fresh300 seed42',config=cfg,arms=records,model=dict(base='original E0 four-stage decoder, skips, head, normalization and losses unchanged',change='replace only TinyViM-S encoder and its channel alignment with official ImageNet-pretrained encoder',no_A0_A1_A6=True),protected_sha256=sources,data_audit=audit,initialization='official ImageNet encoder; E0 four-stage decoder seed42; alignment seed43; fresh initialization, no HAM segmentation weights',contract='ISIC2018 deduplicated train1780 val258 test518; 300 epochs; batch8; FP32; 66900 updates; validation minimum loss earliest tie; fixed-best test518 noTTA and mirror4',queue=dict(status='waiting_for_existing_queues',after='ISIC A1 encoder workers, HAM A016 workers and MISSING3 continuation queue complete'),storage_required_bytes=needed)
 write(RUN/'protocol.json',p)
 for f,h in sources.items():
  src=Path(f);dst=RUN/'source_snapshot'/src.relative_to('/');dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);assert sha(dst)==h
 write(RUN/'prepare_complete.json',dict(passed=True,arms=list(ARMS),protocol_sha256=sha(RUN/'protocol.json')))
 print('PREPARED E0 encoder-only controls:',', '.join(ARMS),flush=True)

def setup_base():
 from operations.isic_a01_20261108 import entry as base
 base.RUN=RUN;base.REF=REF;base.UUIDS=UUIDS;base.imports=imports;base.verify=verify
 return base

def verify():
 p=read(RUN/'protocol.json')
 for f,h in p['protected_sha256'].items():assert sha(f)==h,'protected source changed: '+f
 assert sha(p['config']['data']['split_file'])==p['config']['data']['split_sha256']
 assert shutil.disk_usage(RUN).free>8*2**30
 return p

def phase(arm,mode,gpu):
 base=setup_base();base.phase(arm,mode,gpu)

def controller(arm,gpu):
 import fcntl
 work=RUN/arm
 with (work/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for mode in ('preflight','train','test'):
   verify();env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=UUIDS[gpu],EXPERIMENT_PHYSICAL_GPU=str(gpu),EXPERIMENT_GPU_UUID=UUIDS[gpu],OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
   with (work/(mode+'.log')).open('x') as log:
    child=subprocess.Popen([PY,'-B','-u',__file__,'phase',arm,mode,str(gpu)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    write(work/'controller_status.json',dict(status=mode,controller_pid=os.getpid(),child_pid=child.pid,gpu=gpu,updated_unix=time.time()));rc=child.wait()
   write(work/(mode+'_exitcode.json'),dict(exitcode=rc,pid=child.pid));assert rc==0,mode+' failed; downstream blocked'
  assert read(work/'completion.json')['complete']
  write(work/'controller_status.json',dict(status='complete',controller_pid=os.getpid(),gpu=gpu))

def wait_existing_queues():
 import time
 reroute=ROOT/'operations/gpu1_queue_reroute_20261108/migration_status.json'
 while True:
  state=read(reroute) if reroute.exists() else {'status':'not_started'}
  if state.get('status')=='complete':break
  if state.get('status')=='failed':raise RuntimeError('GPU1 queue reroute failed; refusing to launch E0: '+str(state))
  write(RUN/'queue_status.json',dict(status='waiting_gpu1_reroute',reroute_state=state.get('status'),updated_unix=time.time()));time.sleep(30)
 old_ops=ROOT/'operations/encoder_swap_3slot_takeover_20261108'
 miss=ROOT/'operations/missing3_resume300_20261107/queue_status.json'
 isic_workers=(2525424,2525428)
 def alive(pid,expected=None):
  try:
   proc=Path('/proc',str(int(pid)))
   cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
   return bool(cmd) and (expected is None or expected in cmd)
  except (OSError,TypeError,ValueError):return False
 while True:
  blockers=[p for p in isic_workers if alive(p,'operations/isic_a1_encoder_20261108/run.py worker')]
  for f in old_ops.glob('state_*.json'):
   d=read(f);pid=d.get('worker_pid')
   if pid and alive(pid,'operations/encoder_swap_3slot_takeover_20261108/controller.py worker'):blockers.append(pid)
  md=read(miss)
  if md.get('status')!='complete':
   pid=md.get('pid')
   if pid and alive(pid,'operations/missing3_resume300_20261107/entry.py controller'):blockers.append(pid)
   else:raise RuntimeError('MISSING3 queue exited before recording completion; refusing to bypass it')
  if not blockers:break
  write(RUN/'queue_status.json',dict(status='waiting_existing_queues',blocked_by=sorted(set(blockers)),updated_unix=time.time()));time.sleep(60)
 # Wait for headroom so preflight cannot compete with another unrelated run.
 while True:
  try:
   rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
   free={r.split(',')[0].strip():int(r.split(',')[1].strip()) for r in rows}
   if all(free.get(u,0)>=12000 for u in UUIDS):break
  except Exception:pass
  write(RUN/'queue_status.json',dict(status='waiting_gpu_memory',free_mib=free if 'free' in locals() else {},updated_unix=time.time()));time.sleep(60)
 write(RUN/'queue_status.json',dict(status='launching',updated_unix=time.time()))
 exitcodes=[]
 for arm in ARMS:
  write(RUN/'queue_status.json',dict(status='running_arm',arm=arm,gpu=0,updated_unix=time.time()))
  with (RUN/arm/'queue.log').open('a') as log:
   p=subprocess.Popen([PY,'-B','-u',__file__,'controller',arm,'0'],cwd=ROOT,start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
   rc=p.wait()
  exitcodes.append(rc)
  if rc!=0:break
 write(RUN/'queue_status.json',dict(status='complete' if len(exitcodes)==len(ARMS) and all(rc==0 for rc in exitcodes) else 'failed',exitcodes=exitcodes,assigned_gpu=0,serial=True,updated_unix=time.time()))

if __name__=='__main__':
 try:
  mode=sys.argv[1]
  if mode=='prepare':prepare()
  elif mode=='controller':controller(sys.argv[2],int(sys.argv[3]))
  elif mode=='phase':phase(sys.argv[2],sys.argv[3],int(sys.argv[4]))
  elif mode=='queue':wait_existing_queues()
 except BaseException as e:
  dest=RUN/(sys.argv[2] if len(sys.argv)>2 and sys.argv[1] in ('phase','controller') else '')
  if dest.exists():write(dest/'failure.json',dict(error=repr(e),pid=os.getpid(),time=time.time()))
  raise

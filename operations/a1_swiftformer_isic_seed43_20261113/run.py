"""A1 encoder replacement fresh300 experiments on deduplicated ISIC2018."""
import os,sys,json,copy,io,time,hashlib,shutil,subprocess,fcntl,traceback
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
OPS=Path(__file__).resolve().parent
RUN=Path('/media/8T/PuMengYu/experiments/skin_segmentation/A1_SWIFTFORMER_ISIC_SEED43_FRESH300_20261113')
PY='/home/PuMengYu/anaconda3/envs/medseg/bin/python'
UUIDS=['GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde','GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b']
ARMS=['swiftformer_s']
ASSIGN={'swiftformer_s':1}
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba'),str(OPS.parent/'encoder_swap_A016_20261107')]
from operations.encoder_swap_A016_20261107.encoder_swap_candidate import build_candidate,original,structure
from operations.a1_swiftformer_isic_seed43_20261113 import entry as common
common.RUN=RUN;common.UUIDS=UUIDS
REF=Path('/media/8T/PuMengYu/experiments/skin_segmentation/ISIC2018_A0_A1_FRESH300_20261108/protocol.json')

def read(p):return json.loads(Path(p).read_text())
def write(p,o):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(o,ensure_ascii=False,indent=2));t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def digest(sd):return common.digest(sd)
def imports(arm):
 import torch,ham_fast_runner as runner
 from source.isic2018 import data
 return torch,runner,data,lambda pretrained:build_candidate(arm+'__A1',pretrained)
common.imports=imports

def verify():
 p=read(RUN/'protocol.json')
 for f,h in p['protected_sha256'].items():assert sha(f)==h,'source drift '+f
 assert sha(p['config']['data']['split_file'])==p['config']['data']['split_sha256']
 assert shutil.disk_usage(RUN.parent).free>8*2**30
 return p
common.verify=verify

def prepare():
 import torch,ham_fast_runner as runner
 from source.isic2018 import data
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not RUN.exists()
 torch.set_num_threads(1);meta=data.verify();parent=read(REF);cfg=copy.deepcopy(parent['config']);cfg['training']['seed']=43
 assert meta['counts']==dict(train=1780,val=258,test=518)
 audit=read(data.DEDUP/'split_audit.json')
 assert audit['checks']['unresolved_review_pairs']==audit['checks']['remaining_multi_active_original_strong_components']==0
 assert sha(data.DEDUP/'split.csv')==audit['split_sha256']==cfg['data']['dedup_split_sha256']
 assert cfg['training']['epochs']==300 and cfg['training']['batch_size']==8 and cfg['training']['total_optimizer_updates']==66900
 RUN.mkdir(parents=True);records={};sizes=[];api=runner.baseline_api()
 for arm in ARMS:
  api.seed_all(43);m=build_candidate(arm+'__A1',True);structure(m,'A1')
  api.seed_all(43);ref=original('A1',False)
  kept={k:v for k,v in m.state_dict().items() if not k.startswith(('encoder.','align.'))}
  base={k:v for k,v in ref.state_dict().items() if not k.startswith(('encoder.','align.'))}
  assert kept.keys()==base.keys() and all(torch.equal(v,base[k]) for k,v in kept.items())
  assert all(q.requires_grad for q in m.parameters())
  initial=digest(m.state_dict());buf=io.BytesIO();torch.save(m.state_dict(),buf);buf.seek(0);m.load_state_dict(torch.load(buf,map_location='cpu',weights_only=False),strict=True)
  assert not torch.cuda.is_initialized();size=sum(t.numel()*t.element_size() for t in m.state_dict().values());sizes.append(size)
  rec=dict(initial_state_sha256=initial,state_bytes=size,parameters=sum(q.numel() for q in m.parameters()),retained_A1_decoder_and_head_exact=True,full_state_strict_roundtrip=True,all_parameters_trainable=True,encoder_loading_audit=m.encoder.loading_audit,assignment_gpu=ASSIGN[arm])
  (RUN/arm).mkdir();write(RUN/arm/'cpu_acceptance.json',dict(passed=True,cuda_initialized=False,**rec));write(RUN/arm/'weight_loading_audit.json',m.encoder.loading_audit);records[arm]=rec
  print('CPU_ACCEPTED',arm,rec['parameters'],flush=True);del m,ref,kept,base,buf
 required=sum(sizes)*8+4*2*518*256*256+2**30
 assert shutil.disk_usage(RUN).free>required+8*2**30,'ISIC storage reserve below 8GiB'
 sources={str(REF):sha(REF)}
 for mod in list(sys.modules.values()):
  f=getattr(mod,'__file__',None)
  if f and str(f).startswith(str(ROOT)) and f.endswith('.py') and Path(f).is_file():sources[str(Path(f).resolve())]=sha(f)
 for f in [Path(__file__),ROOT/'operations/encoder_swap_A016_20261107/encoder_swap_candidate.py',ROOT/'operations/isic_a01_20261108/entry.py',ROOT/'source/bottleneck_mamba/final_test_pipeline.py',ROOT/'operations/nnunet_mirror_validation/run_validation.py',ROOT/'operations/nnunet_mirror_validation/mirror_core.py']:sources[str(f)]=sha(f)
 for arm in ARMS:
  for v in records[arm]['encoder_loading_audit'].values():
   if isinstance(v,str) and Path(v).is_file():sources[v]=sha(v)
 protocol=dict(identity='A1 with official SwiftFormer-S pretrained encoder, fixed deduplicated ISIC2018 split, fresh300 training seed43',config=cfg,arms=records,data_audit=audit,protected_sha256=sources,initialization='Original A1 decoder/head fresh initialization under training seed43; encoder strict official SwiftFormer-S classifier pretrained weights; new 1x1 alignment initialized under training seed43; all parameters trainable; no HAM segmentation checkpoint',contract='ISIC2018 deduplicated train1780 val258 test518; 300 epochs, batch8, 66900 optimizer updates, FP32; earliest validation minimum-loss checkpoint; fixed-best test518 noTTA and mirror4; test not used for selection',scheduling=dict(user_authorized_immediate_shared_gpu=True,at_most_one_added_job_per_gpu=True,existing_tasks_preserved=True,admission_lock=str(ROOT/'operations/encoder_swap_3slot_takeover_20261108/admission.lock'),free_vram_gate_mib=8192,max_compute_processes_per_gpu=4),storage_required_bytes=required)
 write(RUN/'protocol.json',protocol)
 for f,h in sources.items():
  dst=RUN/'source_snapshot'/Path(f).relative_to('/');dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dst);assert sha(dst)==h
 write(RUN/'prepare_complete.json',dict(passed=True,arms=ARMS,protocol_sha256=sha(RUN/'protocol.json'),dedup_split_sha256=cfg['data']['split_sha256']))

def phase(mode,arm,gpu):
 p=verify();common.RUN=RUN;common.UUIDS=UUIDS;common.verify=verify;common.imports=imports
 common.phase(arm,mode,gpu)

def launch(gpu,arm,mode):
 guard=Path('/home/PuMengYu/experiments/skin_segmentation/operations/encoder_swap_3slot_takeover_20261108/admission.lock')
 while True:
  lock=guard.open('a');fcntl.flock(lock,fcntl.LOCK_EX)
  try:
   out=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
   pids=[int(x.split(',')[1]) for x in out.splitlines() if x.startswith(UUIDS[gpu])]
   stats=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader'],text=True)
   free=int(next(x.split(',')[1].strip().split()[0] for x in stats.splitlines() if x.startswith(UUIDS[gpu])))
   if len(pids)<4 and free>=8192:break
  finally:
   if not (len(pids)<4 and free>=8192):lock.close()
  write(RUN/(arm+'_waiting.json'),dict(status='waiting_for_gpu_admission',gpu=gpu,compute_pids=pids,free_mib=free,updated_unix=time.time()));time.sleep(30)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=UUIDS[gpu],EXPERIMENT_PHYSICAL_GPU=str(gpu),EXPERIMENT_GPU_UUID=UUIDS[gpu],OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 log=(RUN/arm/(mode+'.log')).open('x');child=subprocess.Popen([PY,'-B','-u',__file__,'phase',mode,arm,str(gpu)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
 write(RUN/arm/(mode+'_status.json'),dict(status=mode,pid=child.pid,gpu=gpu,started_unix=time.time()))
 limit=time.time()+180
 while child.poll() is None:
  out=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
  if child.pid in [int(x.split(',')[1]) for x in out.splitlines() if x.startswith(UUIDS[gpu])]:break
  if time.time()>limit:child.terminate();raise TimeoutError('phase did not become GPU resident')
  time.sleep(1)
 fcntl.flock(lock,fcntl.LOCK_UN);lock.close();rc=child.wait();log.close();write(RUN/arm/(mode+'_exitcode.json'),dict(exitcode=rc,pid=child.pid,finished_unix=time.time()));assert rc==0,arm+' '+mode+' failed'

def worker(gpu):
 for arm in [a for a in ARMS if ASSIGN[a]==gpu]:
  for mode in ['preflight','train','test']:
   verify();launch(gpu,arm,mode)
  assert read(RUN/arm/'completion.json')['complete']
 write(RUN/f'worker_gpu{gpu}_complete.json',dict(complete=True,gpu=gpu))

if __name__=='__main__':
 try:
  if sys.argv[1]=='prepare':prepare()
  elif sys.argv[1]=='worker':worker(int(sys.argv[2]))
  elif sys.argv[1]=='phase':phase(sys.argv[2],sys.argv[3],int(sys.argv[4]))
 except BaseException as e:
  dest=RUN/(sys.argv[3] if len(sys.argv)>3 and sys.argv[1]=='phase' else 'failure.json')
  if RUN.exists():write(dest/'failure.json' if dest.is_dir() else RUN/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),pid=os.getpid(),time=time.time()))
  raise

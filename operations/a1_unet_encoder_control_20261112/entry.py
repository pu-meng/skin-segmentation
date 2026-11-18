import copy, fcntl, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
OPS=ROOT/'operations/a1_unet_encoder_control_20261112'
RUN=Path('/media/8T/PuMengYu/experiments/skin_segmentation/ISIC2018_A1_UNetEncoder_Control_FRESH300_20261112')
PY='/home/PuMengYu/anaconda3/envs/medseg/bin/python'
UUIDS=['GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde','GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b']
REF=Path('/media/8T/PuMengYu/experiments/skin_segmentation/ISIC2018_A0_A1_FRESH300_20261108/protocol.json')
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba'),str(ROOT/'operations/encoder_swap_A016_20261107')]

from operations.encoder_swap_A016_20261107.encoder_swap_candidate import original
from operations.isic_a01_20261108 import entry as common
common.RUN=RUN
common.UUIDS=UUIDS


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');t.replace(p)

def build(pretrained=True):
    import torch
    from torch import nn
    from source.comparison.unet2d import DoubleConv
    model=original('A1',False)
    class PlainUNetEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks=nn.ModuleList([DoubleConv(3,32),DoubleConv(32,64),DoubleConv(64,128),DoubleConv(128,256),DoubleConv(256,512)])
            self.pool=nn.MaxPool2d(2)
            self.loading_audit={'encoder':'plain UNet2D convolutional encoder','source':'source/comparison/unet2d.py','pretrained':False,'features':'levels 1-4: [64@128,128@64,256@32,512@16]','input_normalization':'kept unchanged from A1+SwiftFormer parent'}
        def forward(self,x):
            feats=[]
            for i,block in enumerate(self.blocks):
                if i: x=self.pool(x)
                x=block(x)
                if i: feats.append(x)
            return feats
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42); encoder=PlainUNetEncoder()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(43)
        align=nn.ModuleList(nn.Conv2d(c,c,1) for c in (64,128,256,512))
    model.encoder=encoder;model.align=align
    assert all(p.requires_grad for p in model.parameters())
    return model

def imports(arm):
    import torch,ham_fast_runner as runner
    from source.isic2018 import data
    return torch,runner,data,lambda pretrained:build(pretrained)
common.imports=imports

def verify():
    p=json.loads((RUN/'protocol.json').read_text())
    for f,h in p['protected_sha256'].items(): assert sha(f)==h,'source drift: '+f
    assert sha(p['config']['data']['split_file'])==p['config']['data']['split_sha256']
    assert shutil.disk_usage(RUN).free>8*2**30
    return p
common.verify=verify

def prepare():
    import torch,ham_fast_runner as runner
    from source.isic2018 import data
    assert not RUN.exists(),'refuse to overwrite existing run root'
    assert os.environ.get('CUDA_VISIBLE_DEVICES','')=='','prepare must remain CPU-only'
    parent=json.loads(REF.read_text());cfg=copy.deepcopy(parent['config'])
    meta=data.verify();audit=json.loads((data.DEDUP/'split_audit.json').read_text())
    assert meta['counts']==cfg['data']['counts']==dict(train=1780,val=258,test=518)
    assert audit['checks']['unresolved_review_pairs']==audit['checks']['remaining_multi_active_original_strong_components']==0
    assert sha(data.DEDUP/'split.csv')==audit['split_sha256']==cfg['data']['dedup_split_sha256']
    assert cfg['training']['epochs']==300 and cfg['training']['seed']==42 and cfg['training']['total_optimizer_updates']==66900
    torch.set_num_threads(1);api=runner.baseline_api();api.seed_all(42);m=build(True)
    # Encoder swap invariant: the entire A1 decoder remains byte-identical to the base.
    api.seed_all(42);base=original('A1',False)
    a={k:v for k,v in m.state_dict().items() if not k.startswith(('encoder.','align.'))}
    b={k:v for k,v in base.state_dict().items() if not k.startswith(('encoder.','align.'))}
    assert a.keys()==b.keys() and all(torch.equal(v,b[k]) for k,v in a.items())
    buf=__import__('io').BytesIO();torch.save(m.state_dict(),buf);buf.seek(0);m.load_state_dict(torch.load(buf,map_location='cpu',weights_only=False),strict=True)
    assert not torch.cuda.is_initialized()
    rec=dict(initial_state_sha256=common.digest(m.state_dict()),parameters=sum(q.numel() for q in m.parameters()),state_bytes=sum(t.numel()*t.element_size() for t in m.state_dict().values()),retained_A1_decoder_and_head_exact=True,full_state_strict_roundtrip=True,all_parameters_trainable=True,encoder_loading_audit=m.encoder.loading_audit)
    del m,base,a,b,buf
    RUN.mkdir(parents=True);(RUN/'A1').mkdir()
    sources={str(REF):sha(REF)}
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py') and Path(f).is_file():sources[str(Path(f).resolve())]=sha(f)
    for f in [Path(__file__),ROOT/'operations/encoder_swap_A016_20261107/encoder_swap_candidate.py',ROOT/'operations/isic_a01_20261108/entry.py',ROOT/'source/comparison/unet2d.py',ROOT/'source/bottleneck_mamba/final_test_pipeline.py',ROOT/'operations/nnunet_mirror_validation/run_validation.py',ROOT/'operations/nnunet_mirror_validation/mirror_core.py']:
        sources[str(f)]=sha(f)
    for n,v in rec['encoder_loading_audit'].items():
        if isinstance(v,str) and Path(v).is_file():sources[str(Path(v).resolve())]=sha(v)
    needed=rec['state_bytes']*8+5*2**30
    assert shutil.disk_usage(RUN).free>needed+8*2**30,'insufficient storage reserve'
    cfg['evaluation']['completion']='300 epochs -> val-minloss fixed best -> CPU strict load -> test518 none and mirror4 -> artifact audits'
    protocol=dict(identity='A1 decoder unchanged + plain UNet2D convolutional encoder, deduplicated ISIC2018 fresh300 seed42',config=cfg,arms={'A1':rec},data_audit=audit,protected_sha256=sources,initialization='A1 decoder/head kept exactly; replace only encoder with standard five-block UNet2D convolutional encoder; use encoder outputs 64@128,128@64,256@32,512@16; preserve A1 input normalization; same-width trainable 1x1 alignment; seed42; no segmentation checkpoint',contract='ISIC2018 dedup train1780 val258 test518; 300 epochs; batch8; 66900 updates; FP32; validation-minloss selection; test not used for selection',scheduling=dict(physical_gpu=1,gpu_uuid=UUIDS[1],shared_gpu_authorized=True,admission_lock=str(ROOT/'operations/encoder_swap_3slot_takeover_20261108/admission.lock'),free_vram_gate_mib=8192,max_compute_processes=4),storage_required_bytes=needed)
    write(RUN/'protocol.json',protocol)
    for f,h in sources.items():
        dst=RUN/'source_snapshot'/Path(f).relative_to('/');dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dst);assert sha(dst)==h
    write(RUN/'A1/cpu_acceptance.json',dict(passed=True,cuda_initialized=False,**rec))
    write(RUN/'prepare_complete.json',dict(passed=True,protocol_sha256=sha(RUN/'protocol.json'),retained_A1_decoder_and_head_exact=True))
    print('CPU_PREPARE_COMPLETE',json.dumps(rec,ensure_ascii=False),flush=True)

def phase(mode):
    p=verify()
    common.RUN=RUN;common.UUIDS=UUIDS;common.verify=verify;common.imports=imports
    common.phase('A1',mode,1)

def launch(mode):
    guard=ROOT/'operations/encoder_swap_3slot_takeover_20261108/admission.lock'
    while True:
        lock=guard.open('a');fcntl.flock(lock,fcntl.LOCK_EX)
        out=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        pids=[int(x.split(',')[1]) for x in out.splitlines() if x.startswith(UUIDS[1])]
        stats=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader'],text=True)
        free=int(next(x.split(',')[1].strip().split()[0] for x in stats.splitlines() if x.startswith(UUIDS[1])))
        if len(pids)<4 and free>=8192:break
        lock.close();write(RUN/'A1/controller_status.json',dict(status='waiting_gpu_admission',gpu=1,compute_pids=pids,free_mib=free,updated_unix=time.time()));time.sleep(30)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=UUIDS[1],EXPERIMENT_PHYSICAL_GPU='1',EXPERIMENT_GPU_UUID=UUIDS[1],OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    log=(RUN/'A1'/f'{mode}.log').open('x')
    child=subprocess.Popen([PY,'-B','-u',str(OPS/'entry.py'),'phase',mode],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(RUN/'A1/controller_status.json',dict(status=mode,controller_pid=os.getpid(),child_pid=child.pid,physical_gpu=1,gpu_uuid=UUIDS[1],prior_compute_pids=pids,free_mib_at_admission=free,updated_unix=time.time()))
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    rc=child.wait();log.close();write(RUN/'A1'/f'{mode}_exitcode.json',dict(exitcode=rc,pid=child.pid,time=time.time()))
    assert rc==0,f'{mode} failed; later phases blocked'

def controller():
    lock=(RUN/'A1/controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for mode in ('preflight','train','test'):
        verify();launch(mode)
    write(RUN/'A1/controller_status.json',dict(status='complete',physical_gpu=1,updated_unix=time.time()))
    print('RUN_COMPLETE',RUN,flush=True)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='phase': phase(sys.argv[2])
    elif len(sys.argv)>1 and sys.argv[1]=='prepare': prepare()
    elif len(sys.argv)>1 and sys.argv[1]=='controller': controller()
    else: raise SystemExit('usage: entry.py prepare|controller|phase MODE')

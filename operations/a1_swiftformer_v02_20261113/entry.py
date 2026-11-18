"""Train A1-swiftformer-v0.2 independently on HAM10000 and ISIC2018."""
import copy, csv, fcntl, hashlib, json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path

ROOT = Path('/home/PuMengYu/experiments/skin_segmentation')
OPS = Path(__file__).resolve().parent
LABEL = 'A1-swiftformer-v0.2'
RUN = Path('/media/8T/PuMengYu/experiments/skin_segmentation/A1-swiftformer-v0.2_HAM_ISIC_FRESH300_20261113')
PY = '/home/PuMengYu/anaconda3/envs/medseg/bin/python'
UUIDS = ['GPU-a87a547f-d856-16b2-a1e5-b5f26d974fde', 'GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b']
HAM_REF = Path('/media/8T/PuMengYu/experiments/skin_segmentation/DecoderDualA_RESUME100to300_20261106/protocol.json')
ISIC_REF = Path('/media/8T/PuMengYu/experiments/skin_segmentation/ISIC2018_A0_A1_FRESH300_20261108/protocol.json')
sys.path[:0] = [str(ROOT), str(ROOT/'source/comparison'), str(ROOT/'source/bottleneck_mamba')]

def read(p): return json.loads(Path(p).read_text())
def write(p, o):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n'); t.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def state_digest(sd):
    h=hashlib.sha256()
    for n,t in sorted(sd.items()): h.update(n.encode()); h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def imports():
    from pretrained_unet2d import prepare_imports
    prepare_imports('mambavision_t'); prepare_imports('tinyvim_s')
    import torch, ham_fast_runner as runner
    from source.a1_swiftformer_v02_20261113.candidate import build_candidate
    return torch,runner,build_candidate

def prepare():
    assert not RUN.exists(), 'refuse to overwrite existing run root'
    assert HAM_REF.is_file() and ISIC_REF.is_file()
    torch,runner,build=imports(); torch.set_num_threads(1)
    ham=read(HAM_REF); isic=read(ISIC_REF)
    for rel,h in ham['source_sha256'].items(): assert sha(ROOT/rel)==h, 'HAM parent source drift: '+rel
    for rel,h in isic['protected_sha256'].items(): assert sha(rel)==h, 'ISIC parent source drift: '+rel
    hcfg=copy.deepcopy(ham['config']); icfg=copy.deepcopy(isic['config'])
    assert hcfg['data']['counts']=={'train':6983,'val':1010,'test':2020}
    assert hcfg['training']['epochs']==300 and hcfg['training']['batch_size']==8
    assert icfg['data']['counts']=={'train':1780,'val':258,'test':518}
    assert icfg['training']['epochs']==300 and icfg['training']['batch_size']==8
    from source.isic2018 import data as isic_data
    isic_audit=isic_data.verify(); assert isic_audit['counts']==icfg['data']['counts']
    api=runner.baseline_api(); records={}
    for ds,cfg in [('HAM10000',hcfg),('ISIC2018',icfg)]:
        api.seed_all(42); m=build(True)
        assert all(p.requires_grad for p in m.parameters())
        sd=state_digest(m.state_dict()); buf=__import__('io').BytesIO(); torch.save(m.state_dict(),buf); buf.seek(0)
        m.load_state_dict(torch.load(buf,map_location='cpu',weights_only=False),strict=True)
        records[ds]=dict(initial_state_sha256=sd,parameters=sum(p.numel() for p in m.parameters()),
                         state_bytes=sum(t.numel()*t.element_size() for t in m.state_dict().values()),
                         model_class=type(m).__name__,encoder_loading_audit=m.encoder.loading_audit)
        del m,buf
    assert records['HAM10000']['initial_state_sha256']==records['ISIC2018']['initial_state_sha256']
    RUN.mkdir(parents=True); (RUN/'HAM10000').mkdir(); (RUN/'ISIC2018').mkdir()
    sources={str(HAM_REF):sha(HAM_REF),str(ISIC_REF):sha(ISIC_REF),str(Path(__file__)):sha(__file__)}
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py') and Path(f).is_file(): sources[str(Path(f).resolve())]=sha(f)
    for item in records['HAM10000']['encoder_loading_audit'].values():
        if isinstance(item,str) and Path(item).is_file(): sources[item]=sha(item)
    for ds,cfg in [('HAM10000',hcfg),('ISIC2018',icfg)]:
        split=Path(cfg['data']['split_file']); assert sha(split)==cfg['data']['split_sha256']
        if ds=='ISIC2018': assert sha(isic_data.CACHE/'cache_complete.json')==cfg['data']['cache_manifest_sha256']
    need=sum(x['state_bytes'] for x in records.values())*10+3*2**30
    assert shutil.disk_usage(RUN).free>need+8*2**30
    protocol=dict(identity=f'{LABEL} fresh300 on deduplicated ISIC2018 and HAM10000',datasets={
        'HAM10000':dict(config=hcfg,parent_protocol=str(HAM_REF),split_sha256=hcfg['data']['split_sha256'],counts=hcfg['data']['counts'],baseline='A1: 94.9322 no-TTA, 95.0459 mirror4 Global Dice'),
        'ISIC2018':dict(config=icfg,parent_protocol=str(ISIC_REF),split_sha256=icfg['data']['split']['sha256'] if isinstance(icfg['data'].get('split'),dict) else icfg['data']['split_sha256'],counts=icfg['data']['counts'],baseline='A1: 90.9956 no-TTA, 91.3286 mirror4 Global Dice')},
        model=records,architecture='SwiftFormer-S + original A1 E2A/E3/head; three repeated D-WPO->upsample/skip/DualStage blocks spanning D3-D2, D2-D1 and D1-D0; no Bottle change.',initialization='SwiftFormer-S official ImageNet classification weights; newly initialized channel alignment and modified decoder modules; no segmentation checkpoint; all trainable',
        training='seed42; fresh 300 epochs; batch8 FP32; inherited dataset-specific AdamW/lr/scheduler/loss/augmentation/sampler/validation rule; all parameters trainable; test excluded from selection',
        evaluation='fixed validation-minloss best; strict CPU load; test no-TTA and mirror4; complete predictions/per-image CSV/summary/visualizations/audit',
        protected_sha256=sources,shared_gpu_authorized=True,scheduling={'HAM10000':{'physical_gpu':0,'uuid':UUIDS[0]},'ISIC2018':{'physical_gpu':1,'uuid':UUIDS[1]},'parallel':True,'per_gpu_project_slots':3,'resource_gate':'each worker waits until assigned GPU free>=3500 MiB and <6 compute processes; real batch8 preflight gates each run'},
        storage_required_bytes=need,created_unix=time.time())
    write(RUN/'protocol.json',protocol)
    for ds in ['HAM10000','ISIC2018']:
        d=RUN/ds; write(d/'cpu_acceptance.json',dict(passed=True,strict_state_roundtrip=True,cuda_initialized=False,**records[ds]))
    for f,h in sources.items():
        dst=RUN/'source_snapshot'/Path(f).relative_to('/'); dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(f,dst); assert sha(dst)==h
    write(RUN/'prepare_complete.json',dict(passed=True,datasets=['HAM10000','ISIC2018'],source_hashes=len(sources),protocol_sha256=sha(RUN/'protocol.json')))
    print('CPU_PREPARE_COMPLETE',json.dumps(records,ensure_ascii=False),flush=True)

def verify():
    p=read(RUN/'protocol.json')
    for f,h in p['protected_sha256'].items(): assert sha(f)==h,'source drift: '+f
    for ds,r in p['datasets'].items(): assert sha(r['config']['data']['split_file'])==r['split_sha256']
    assert shutil.disk_usage(RUN).free>8*2**30
    return p

def data_loaders(ds,cfg,api):
    if ds=='HAM10000':
        from ham_fast_pipeline import make_fast_loaders
        return make_fast_loaders(cfg,api,workers=2,validation_batch=8)
    from source.isic2018 import data
    return data.loaders(cfg,api)

def phase(ds,kind,gpu):
    p=verify(); cfg=p['datasets'][ds]['config']; torch,runner,build=imports(); api=runner.baseline_api()
    api.seed_all(42); torch.set_num_threads(4); model=build(kind!='test')
    work=RUN/ds; run=work/'run'
    if kind=='test': return test(ds,gpu,p,model,torch,runner,api)
    stage=work/('preflight' if kind=='preflight' else 'run'); stage.mkdir(exist_ok=True)
    if kind=='train': assert read(work/'preflight/acceptance.json')['status']=='passed'
    torch.cuda.set_device(0); model.cuda(); loaders,generator=data_loaders(ds,cfg,api)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['training']['lr'],weight_decay=cfg['training']['weight_decay'],betas=tuple(cfg['training']['betas']),eps=cfg['training']['eps'])
    criterion=api.BceDiceLoss().cuda(); device=torch.device('cuda:0')
    info=dict(model=LABEL,dataset=ds,seed=42,pid=os.getpid(),physical_gpu=gpu,gpu_uuid=UUIDS[gpu],
              split_sha256=p['datasets'][ds]['split_sha256'],protocol_sha256=sha(RUN/'protocol.json'),source_sha256=p['protected_sha256'],
              parameters=sum(q.numel() for q in model.parameters()),fresh_start=True,test_data_accessed=False)
    write(stage/'provenance.json',info); write(stage/'weight_loading_audit.json',model.encoder.loading_audit)
    try:
        if kind=='preflight':
            proof=runner.preflight(model,loaders,criterion,opt,device,stage,[])
            proof.update(dataset=ds,physical_gpu=gpu,gpu_uuid=UUIDS[gpu],protocol_sha256=info['protocol_sha256'],test_data_accessed=False)
            write(stage/'acceptance.json',proof); print('REAL_BATCH8_PREFLIGHT_PASS',ds,json.dumps(proof),flush=True)
        elif kind=='train':
            first=True; original=opt.step
            def observed_step(self,*args,**kwargs):
                nonlocal first
                if first:
                    grads=[q.grad for q in model.parameters() if q.requires_grad]
                    assert len(grads)==sum(q.requires_grad for q in model.parameters()) and all(q is not None and torch.isfinite(q).all() for q in grads)
                    probe=next((n,q) for n,q in model.named_parameters() if q.requires_grad and q.grad is not None and q.grad.abs().max()>0)
                    name,q=probe; before=q.detach().clone()
                result=original(*args,**kwargs)
                if first:
                    torch.cuda.synchronize(); delta=float((q-before).abs().max()); assert delta>0
                    write(work/'first_optimizer_update.json',dict(dataset=ds,pid=os.getpid(),physical_gpu=gpu,gpu_uuid=UUIDS[gpu],
                         finite_gradients=True,updated_parameter=name,max_delta=delta,batch_size=8,epoch=1,step=1,test_data_accessed=False))
                    print('FIRST_FORMAL_OPTIMIZER_UPDATE',ds,name,delta,flush=True); first=False
                return result
            import types
            opt.step=types.MethodType(observed_step,opt)
            runner.train(model,loaders,criterion,opt,device,run,cfg,info,generator,resume=False)
            audit=read(run/'artifact_audit.json')
            assert audit['training_complete'] and audit['validation_artifacts_complete'] and audit['epochs']==300
            print('TRAINING_VALIDATION_COMPLETE',ds,flush=True)
    finally:
        for loader in loaders.values():
            if hasattr(loader,'close'): loader.close()
            elif getattr(loader,'_iterator',None) is not None: loader._iterator._shutdown_workers()

def test(ds,gpu,p,model,torch,runner,api):
    from torch.utils.data import DataLoader
    from final_test_pipeline import fixed_best_record,validate_outputs
    from operations.nnunet_mirror_validation.run_validation import Predictor
    work=RUN/ds; fixed=fixed_best_record(work/'run'); assert fixed['history_rows']==300
    out=work/'final_test'; out.mkdir(exist_ok=True); cp=out/'fixed_best.pth'; shutil.copy2(fixed['checkpoint'],cp)
    assert sha(cp)==fixed['checkpoint_sha256']; model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False),strict=True)
    write(out/'cpu_load_audit.json',dict(strict=True,checkpoint_sha256=sha(cp),selected_epoch=fixed['selected_epoch'],cuda_initialized=torch.cuda.is_initialized()))
    if ds=='HAM10000':
        from project_paths import PATHS
        datadir=Path(PATHS['active_split']).parent; dataset=api.HAMDataset(datadir,'test'); expected=2020
    else:
        from source.isic2018 import data
        dataset=data.ISICDataset(data.CACHE,'test'); expected=518
    ids={r['image_id'] for r in dataset.rows}; assert len(dataset)==len(ids)==expected
    loader=DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0)
    torch.cuda.set_device(0); model.cuda().eval(); results={}
    for mode in ['none','mirror4']:
        target=out/mode; target.mkdir(exist_ok=False); pred=Predictor(model,mode=='mirror4').eval()
        result=api.evaluate(pred,loader,api.BceDiceLoss().cuda(),torch.device('cuda:0'),target)
        assert len(pred.times)==expected and pred.verify_remaining==0
        result.update(dataset=ds,model=LABEL,split='test',seed=42,selected_epoch=fixed['selected_epoch'],checkpoint=str(cp),
                      checkpoint_sha256=sha(cp),split_sha256=p['datasets'][ds]['split_sha256'],tta=mode=='mirror4',
                      training_epochs=300,test_used_for_checkpoint_selection=False,physical_gpu=gpu,gpu_uuid=UUIDS[gpu])
        audit=validate_outputs(target,ids,result); assert audit['complete']; write(target/'summary.json',result); write(target/'artifact_audit.json',audit)
        pred.close(); results[mode]=result; print('TEST_COMPLETE',ds,mode,result['global'],flush=True)
    write(work/'completion.json',dict(complete=True,epochs=300,selected_epoch=fixed['selected_epoch'],test_images_per_mode=expected,results=results))
    write(work/'controller_status.json',dict(status='complete',physical_gpu=gpu,pid=os.getpid(),updated_unix=time.time()))

def resource_ready(gpu):
    import csv as _csv
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    pids=[int(x.split(',')[1]) for x in rows.splitlines() if x.startswith(UUIDS[gpu])]
    stats=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader'],text=True)
    free=int(next(x.split(',')[1].strip().split()[0] for x in stats.splitlines() if x.startswith(UUIDS[gpu])))
    return len(pids)<6 and free>=3500,pids,free

def active_project_jobs(ds):
    roots=[
        RUN.parent/'A1-swiftformer-v0.1_HAM_ISIC_FRESH300_20261113'/ds/'controller_status.json',
        RUN/ds/'controller_status.json',
        RUN.parent/'A1_ESBD_SPLIT_HAM_ISIC_FRESH300_20261113'/ds,
        RUN.parent/'A1_ESBD_INCREMENT_HAM_ISIC_FRESH300_20261113'/ds,
    ]
    files=[]
    for item in roots:
        if item.is_dir(): files.extend(item.glob('*/controller_status.json'))
        elif item.is_file(): files.append(item)
    active=[]
    for path in files:
        if not path.is_file(): continue
        state=read(path)
        pid=state.get('child_pid')
        if state.get('status') in ('preflight','train','test') and pid and Path(f'/proc/{pid}').exists():
            active.append((str(path),int(pid)))
    return active

def controller(ds,gpu):
    work=RUN/ds
    with (work/'controller.lock').open('a') as local_lock:
        fcntl.flock(local_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for kind in ['preflight','train','test']:
            verify()
            env=os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=UUIDS[gpu],EXPERIMENT_PHYSICAL_GPU=str(gpu),EXPERIMENT_GPU_UUID=UUIDS[gpu],OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
            with (work/(kind+'.log')).open('x') as log:
                child=subprocess.Popen([PY,'-B','-u',__file__,'phase',ds,kind,str(gpu)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                write(work/'controller_status.json',dict(status=kind,controller_pid=os.getpid(),child_pid=child.pid,physical_gpu=gpu,gpu_uuid=UUIDS[gpu],updated_unix=time.time()))
                rc=child.wait()
            write(work/(kind+'_exitcode.json'),dict(exitcode=rc,pid=child.pid,gpu=gpu,finished_unix=time.time()))
            if rc != 0: raise RuntimeError(f'{ds} {kind} failed with exit code {rc}; see {work / (kind+".log")}')
        assert read(work/'completion.json')['complete']
        write(work/'controller_status.json',dict(status='complete',physical_gpu=gpu,pid=os.getpid(),updated_unix=time.time()))

def wait_for_gpu(ds, gpu, poll=120):
    if os.environ.get("SKIP_WAIT") == "1":
        ok, pids, free = resource_ready(gpu)
        return pids, free
    while True:
        ok,pids,free=resource_ready(gpu)
        if ok: return pids,free
        write(RUN/ds/'queue_status.json',dict(status='waiting_gpu',gpu=gpu,free_mib=free,compute_pids=pids,need_free_mib=8192,updated_unix=time.time()))
        time.sleep(poll)

def dataset_queue(ds, gpu):
    assert (ds, gpu) in [('HAM10000', 0), ('ISIC2018', 1)]
    done=RUN/ds/'completion.json'
    if done.exists() and read(done).get('complete'):
        return
    write(RUN/ds/'queue_status.json',dict(status='queued',dataset=ds,gpu=gpu,uuid=UUIDS[gpu],updated_unix=time.time()))
    while True:
        active=active_project_jobs(ds)
        if len(active)<3: break
        write(RUN/ds/'queue_status.json',dict(status='waiting_project_slot',dataset=ds,gpu=gpu,active=active,slot_limit=3,updated_unix=time.time()))
        time.sleep(10)
    pids,free=wait_for_gpu(ds,gpu)
    write(RUN/ds/'queue_status.json',dict(status='resource_gate_passed',dataset=ds,gpu=gpu,uuid=UUIDS[gpu],compute_pids=pids,free_mib=free,updated_unix=time.time()))
    controller(ds,gpu)

if __name__=='__main__':
    try:
        if sys.argv[1]=='prepare': prepare()
        elif sys.argv[1]=='controller': controller(sys.argv[2],int(sys.argv[3]))
        elif sys.argv[1]=='queue': dataset_queue(sys.argv[2],int(sys.argv[3]))
        elif sys.argv[1]=='phase': phase(sys.argv[2],sys.argv[3],int(sys.argv[4]))
    except BaseException as exc:
        if RUN.exists():
            ds=sys.argv[2] if len(sys.argv)>2 and sys.argv[1] in ('controller','phase','queue') else 'prepare'
            target=RUN/ds if ds in ('HAM10000','ISIC2018') else RUN
            write(target/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),pid=os.getpid(),time=time.time()))
        raise

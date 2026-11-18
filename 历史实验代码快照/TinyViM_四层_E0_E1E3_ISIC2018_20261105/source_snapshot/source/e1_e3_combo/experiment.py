"""Independent HAM/ISIC E1+E3 experiments using existing train/evaluation math."""
from pathlib import Path
import argparse,copy,csv,fcntl,json,os,shutil,signal,sys,time
from datetime import datetime
import torch
from candidate import ROOT,build_candidate
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
from pretrained_unet2d import prepare_imports,sha256_file
prepare_imports('mambavision_t');prepare_imports('tinyvim_s')
import ham_fast_runner as runner
from ham_fast_pipeline import make_fast_loaders
from experiment_v1 import verify_v1_runtime
from final_test_pipeline import fixed_best_record,training_ready,validate_outputs,ProgressLoader
from source.isic2018 import data as isic
HERE=Path(__file__).parent
ROOTS={'HAM':ROOT/'log/TinyViM_E1E3_HAM_20261105','ISIC':ROOT/'log/TinyViM_四层_E0_E1E3_ISIC2018_20261105'}
ARMS={'HAM':['E1_E3'],'ISIC':['E1_E3','E0']}
E0=ROOT/'log/TinyViM_四层对应Decoder_基础_HAM_20261104/protocol.json'
ISIC=ROOT/'log/ISIC2018_瓶颈对比_20261031/protocol.json'
LABELS={'E1_E3':'E1＋E3：D-WPO＋FDCG-S','E0':'E0：四层Decoder基线'}

def write(path,value):runner.write_json(path,value)

def source_hashes():
    paths=set(json.loads(E0.read_text())['source_sha256'])
    paths.update(str(p.relative_to(ROOT)) for p in HERE.glob('*.py'))
    paths.update(['source/three_module_impl_20261104/candidate.py','source/three_module_impl_20261104/official_wave.py',
                  'source/three_module_e23_20261104/candidate.py','source/isic2018/data.py'])
    return {p:sha256_file(ROOT/p) for p in sorted(paths)}

def prepare(dataset):
    runner.verify_runtime=verify_v1_runtime
    runtime=verify_v1_runtime()
    if dataset=='HAM':
        config=copy.deepcopy(runner.validate_protocol(runner.PROTOCOL,'tinyvim_s',check_cache=True))
        parent=json.loads(E0.read_text())
        for k in ['data','training','preprocessing','evaluation']:assert config[k]==parent[k],k
        parent_path=E0
    else:
        parent=json.loads(ISIC.read_text());config=copy.deepcopy(parent);parent_path=ISIC
        # Reuse the completed content-duplicate review, not merely image IDs/SHA.
        meta=isic.verify();audit=json.loads((isic.DEDUP/'split_audit.json').read_text())
        assert audit['checks']['unresolved_review_pairs']==0
        assert audit['checks']['remaining_multi_active_original_strong_components']==0
        assert sha256_file(isic.DEDUP/'split.csv')==audit['split_sha256']==config['data']['dedup_split_sha256']
        assert sha256_file(isic.DEDUP/'excluded_images.csv')==audit['excluded_csv_sha256']
        assert sha256_file(isic.DEDUP/'source_inventory.csv')==audit['source_inventory_sha256']
        policy=json.loads((isic.DEDUP/'exclusion_policy.json').read_text())
        assert sha256_file(Path(policy['evidence']))==audit['decision_sha256']
        assert meta['split_sha256']==config['data']['split_sha256']
        assert sha256_file(isic.CACHE/'cache_complete.json')==config['data']['cache_manifest_sha256']
        assert meta['counts']=={'train':1780,'val':258,'test':518}
        init=config['initialization'];assert sha256_file(Path(init['ImageNet_weight']))==init['ImageNet_sha256']
        assert config['training']['updates_per_epoch']==223 and config['training']['total_optimizer_updates']==66900
    root=ROOTS[dataset];root.mkdir(exist_ok=True)
    for arm in ARMS[dataset]:config['models'][arm]={'run_dir':str(root/arm/'run'),'initialization':'official ImageNet encoder, fresh decoder42/alignment43 and unchanged module seeds42; no HAM segmentation checkpoint'}
    protocol=dict(version='four_stage_E1_E3_'+dataset,dataset=dataset,parent_protocol=str(parent_path),parent_protocol_sha256=sha256_file(parent_path),
        data=config['data'],training=config['training'],preprocessing=config['preprocessing'],evaluation=config['evaluation'],
        arms=ARMS[dataset],source_sha256=source_hashes(),seed=42,test_used_for_model_selection=False,
        model=dict(E0='original four-stage baseline',E1_E3='original E1 replaces only D3 DoubleConv with D-WPO; original E3 CompleteModule refines S0 before D0 concat; other original stages/skips/up/head retained',
                   E1_initialization='exact original E1 DStage seed42',E3_initialization='exact original E3 enhance seed42',
                   E13_additive_version_not_used=True,E2_HSA_not_used=True,HAM_segmentation_weights_loaded=False),
        completion='300 epochs -> minimum validation loss fixed best -> independent predictions/metrics/summary/report/12 visualizations -> audit -> canonical table')
    path=root/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol,'protocol drift'
    else:write(path,protocol)
    for rel,digest in protocol['source_sha256'].items():
        dest=root/'source_snapshot'/rel
        if dest.exists():assert sha256_file(dest)==digest,rel
        else:dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
    write(root/'runtime_validation_audit.json',runtime)
    return config

def aggregate(dataset,arm):
    root=ROOTS[dataset];work=root/arm;summary=work/'test/summary.json';audit=work/'test_artifact_audit.json'
    if not audit.exists() or not json.loads(audit.read_text()).get('complete'):return
    d=json.loads(summary.read_text());m=d['per_image_mean']
    row=f"| {LABELS[arm]} | {m['dice']*100:.4f} | {m['iou']*100:.4f} | {m['hd95_pixels']:.4f} | {d['selected_epoch']} | {d['parameters']/1e6:.3f} | [summary](../{summary.relative_to(ROOT)}) |"
    with (ROOT/'log/E1_E3_report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=ROOT/'研究记录/实验结果汇总.md';text=path.read_text()
        anchor='<a id="four-stage-ham"></a>' if dataset=='HAM' else '<a id="four-stage-isic"></a>'
        start=text.index(anchor);end=text.find('\n<a id=',start+len(anchor));end=len(text) if end<0 else end
        part=text[start:end];lines=part.splitlines();matches=[i for i,l in enumerate(lines) if l.startswith('| '+LABELS[arm]+' |')]
        assert len(matches)==1,(dataset,arm,'reserved row missing')
        lines[matches[0]]=row;path.write_text(text[:start]+'\n'.join(lines)+'\n'+text[end:])

def test(dataset,arm,config):
    work=ROOTS[dataset]/arm;stage=work/'run';assert training_ready(stage)
    fixed=fixed_best_record(stage);assert fixed['history_rows']==300
    model=build_candidate(pretrained=False,arm=arm)
    model.load_state_dict(torch.load(fixed['checkpoint'],map_location='cpu',weights_only=True),strict=True)
    assert not torch.cuda.is_initialized(),'strict test loading must precede CUDA initialization'
    write(work/'cpu_test_load_audit.json',dict(strict_load_passed=True,cuda_initialized_during_load=False,**fixed))
    api=runner.baseline_api();from torch.utils.data import DataLoader
    cache=Path(config['data']['split_file']).parent
    ds=api.HAMDataset(cache,'test') if dataset=='HAM' else isic.ISICDataset(isic.CACHE,'test')
    count=2020 if dataset=='HAM' else 518;assert len(ds)==count and not ds.training
    loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=2,pin_memory=True,multiprocessing_context='spawn')
    output=work/'test';assert not output.exists(),'Refusing test overwrite'
    torch.set_num_threads(4);torch.cuda.set_device(0);device=torch.device('cuda:0');model=model.to(device).eval();criterion=api.BceDiceLoss().to(device)
    state=dict(arm=arm,dataset=dataset,status='testing',total_images=count,completed_images=0,pid=os.getpid(),**fixed)
    write(work/'test_status.json',state);start=time.monotonic()
    result=api.evaluate(model,ProgressLoader(loader,work/'test_status.json',state),criterion,device,output)
    result.update(fixed)
    result.update(dataset='HAM10000' if dataset=='HAM' else 'ISIC2018',model=arm,split='test',seed=42,elapsed_seconds=time.monotonic()-start,test_used_for_checkpoint_selection=False)
    write(output/'summary.json',result)
    (output/'report.txt').write_text(f"{dataset} {LABELS[arm]}；固定验证损失best第{fixed['selected_epoch']}轮；独立测试{count}张\nDice {result['per_image_mean']['dice']*100:.4f}%；IoU {result['per_image_mean']['iou']*100:.4f}%；HD95 {result['per_image_mean']['hd95_pixels']:.4f}像素。测试不参与选择权重。\n")
    write(work/'test_artifact_audit.json',validate_outputs(output,{r['image_id'] for r in ds.rows},result))
    write(work/'test_status.json',dict(state,status='test_complete',completed_images=count,finished_at=datetime.now().astimezone().isoformat()))
    aggregate(dataset,arm);print('TEST_COMPLETE',dataset,arm,flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=['HAM','ISIC'],required=True);parser.add_argument('--arm',choices=['E0','E1_E3'],required=True);parser.add_argument('--mode',choices=['prepare','preflight','train','test'],required=True);args=parser.parse_args()
    assert args.arm in ARMS[args.dataset]
    config=prepare(args.dataset)
    if args.mode=='prepare':return
    work=ROOTS[args.dataset]/args.arm;work.mkdir(exist_ok=True)
    with (work/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.mode=='test':test(args.dataset,args.arm,config);return
        for name in ['cpu_E1_E3_acceptance.json','gpu_E1_E3_acceptance.json']:assert json.loads((HERE/name).read_text())['passed']
        stage=work/('preflight' if args.mode=='preflight' else 'run');stage.mkdir(exist_ok=True)
        pp=ROOTS[args.dataset]/'protocol.json'
        if args.mode=='train':
            proof=json.loads((work/'preflight/preflight_status.json').read_text());assert proof['status']=='passed' and proof['protocol_sha256']==sha256_file(pp)
            assert not any((stage/n).exists() for n in ['status.json','checkpoints','history.csv'])
        api=runner.baseline_api();api.seed_all(42);torch.set_num_threads(4);torch.cuda.set_device(0);device=torch.device('cuda:0')
        model=build_candidate(pretrained=True,arm=args.arm).to(device)
        info=dict(model=args.arm,dataset=args.dataset,seed=42,pid=os.getpid(),fresh_start=True,test_data_accessed=False,parameters=sum(p.numel() for p in model.parameters()),physical_gpu=int(os.environ['EXPERIMENT_PHYSICAL_GPU']),process_cuda_index=0,protocol_file=str(pp),protocol_sha256=sha256_file(pp),split_sha256=config['data']['split_sha256'],training=config['training'],source_sha256=source_hashes(),engineering=dict(runner='unchanged ham_fast_runner',workers=2,validation_batch=8))
        write(stage/'provenance.json',info);write(stage/'weight_loading_audit.json',model.encoder.loading_audit)
        loaders={}
        try:
            loaders,generator=make_fast_loaders(config,api,workers=2,validation_batch=8) if args.dataset=='HAM' else isic.loaders(config,api)
            criterion=api.BceDiceLoss().to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8)
            signal.signal(signal.SIGUSR1,runner.request_stop)
            if args.mode=='preflight':
                watched=list(model.decoder.stages[3].parameters())+list(model.decoder.enhance.parameters()) if args.arm=='E1_E3' else list(model.parameters())[:1]
                before=[p.detach().cpu().clone() for p in watched]
                report=runner.preflight(model,loaders,criterion,opt,device,stage,[])
                changed=sum(not torch.equal(a,p.detach().cpu()) for a,p in zip(before,watched));assert changed==len(watched),(changed,len(watched))
                report.update(protocol_sha256=info['protocol_sha256'],split_sha256=info['split_sha256'],watched_parameter_tensors=len(watched),watched_parameter_tensors_updated=changed,test_loader_constructed=False)
                write(stage/'preflight_status.json',report);print('REAL_PREFLIGHT_PASS',args.dataset,args.arm,json.dumps(report),flush=True)
            else:
                runner.train(model,loaders,criterion,opt,device,stage,config,info,generator,resume=False)
                assert json.loads((stage/'status.json').read_text())['epoch_completed']==300
                f=stage/'validation/summary.json';s=json.loads(f.read_text());s.update(dataset='HAM10000' if args.dataset=='HAM' else 'ISIC2018',model=args.arm);write(f,s)
                (stage/'validation/report.txt').write_text(f"{args.dataset} {args.arm}；300轮，最小验证损失选best第{s['selected_epoch']}轮；完整指标见summary.json。\n")
        except BaseException as e:write(stage/'failure.json',dict(error=repr(e),pid=os.getpid(),mode=args.mode));raise
        finally:
            for loader in loaders.values():
                if hasattr(loader,'close'):loader.close()
                elif getattr(loader,'_iterator',None) is not None:loader._iterator._shutdown_workers()
if __name__=='__main__':main()

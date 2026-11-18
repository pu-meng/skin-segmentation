"""Isolated three-epoch V2 CUDA-equivalent engineering trial.

All outputs go to --root; original modules/checkpoints remain read-only.
The final test is an engineering pilot result, never a 300-epoch comparison.
"""
from pathlib import Path
from datetime import datetime
import argparse
import copy
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
PROJECT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(PROJECT),str(PROJECT/'source/comparison'),str(Path(__file__).parent)]
import torch
from torch import nn
from torch.utils.data import DataLoader
from experiment_v1 import verify_v1_runtime
from experiment_unet_v2 import build_model as original_model
import ham_fast_runner as existing
from ham_fast_pipeline import make_fast_loaders
from pretrained_unet2d import sha256_file
from project_paths import PATHS
from final_test_pipeline import validate_outputs


def write(path,value):
    existing.write_json(path,value)


def build(fast=True):
    model=original_model('unet_v2')
    if fast:
        from Models.TinyViM.mamba_bottleneck_v2_cuda_pilot import BottleneckMambaV2CudaPilot
        previous=model.encoders[-1][1]
        replacement=BottleneckMambaV2CudaPilot(512,16,seed=44)
        replacement.load_state_dict(previous.state_dict(),strict=True)
        model.encoders[-1][1]=replacement
    return model


def close(loaders):
    for loader in loaders.values():
        if hasattr(loader,'close'):loader.close()


def protection():
    names=['Models/TinyViM/mamba_bottleneck_v1.py','Models/TinyViM/mamba_bottleneck_v2.py',
           'source/bottleneck_mamba/experiment_v1.py','source/bottleneck_mamba/experiment_extra.py',
           'source/bottleneck_mamba/experiment_unet_v2.py','source/bottleneck_mamba/final_test_pipeline.py',
           'source/comparison/ham_fast_runner.py','source/comparison/ham_fast_pipeline.py',
           'source/comparison/pretrained_unet2d.py','source/comparison/unet2d.py']
    for name in ['瓶颈Mamba_版本1','瓶颈Mamba_版本2','UNet2D_瓶颈Mamba_版本1']:
        names.extend(['log/'+name+'/run/checkpoints/best.pth','log/'+name+'/run/validation/summary.json'])
    hashes={n:sha256_file(PROJECT/n) for n in names}
    status=json.loads((PROJECT/'log/UNet2D_瓶颈Mamba_版本2/run/status.json').read_text())
    return {'sha256':hashes,'active_training':{k:status.get(k) for k in ['pid','epoch','epoch_completed','step','optimizer_updates']},
            'gpu_state':subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True),
            'time':datetime.now().astimezone().isoformat()}


def benchmark(config,api,root,device):
    api.seed_all(42)
    source=build(False)
    initial={n:v.detach().cpu().clone() for n,v in source.state_dict().items()}
    del source
    loaders,_=make_fast_loaders(config,api,workers=2,validation_batch=8)
    x,y,ids,_=next(iter(loaders['train']));x=x.to(device);y=y.to(device)
    close(loaders)
    reports={}
    for fast in [False,True,True,False]:
        name='cuda_equivalent' if fast else 'explicit_reference'
        model=build(fast);model.load_state_dict(initial,strict=True);model=model.to(device).train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.01)
        criterion=api.BceDiceLoss().to(device)
        samples=[];losses=[]
        torch.cuda.reset_peak_memory_stats(device)
        for index in range(4):
            torch.cuda.synchronize();start=time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss=criterion(model(x),y)
            if not torch.isfinite(loss):raise ValueError('Nonfinite benchmark loss')
            loss.backward();optimizer.step();torch.cuda.synchronize()
            elapsed=time.perf_counter()-start
            if index:samples.append(elapsed)
            losses.append(float(loss.detach()))
        reports.setdefault(name,[]).append({'seconds':samples,'losses':losses,
            'peak_allocated_mib':torch.cuda.max_memory_allocated(device)/2**20})
        del model,optimizer,criterion;torch.cuda.empty_cache()
    values={k:statistics.median([v for rep in reps for v in rep['seconds']]) for k,reps in reports.items()}
    result={'scope':'fixed real train batch; FP32 full forward/backward/AdamW; input already on GPU; one warmup + three timed steps per round; ABBA two rounds per backend',
            'image_ids':list(ids),'raw':reports,'median_step_seconds':values,
            'speedup':values['explicit_reference']/values['cuda_equivalent'],
            'shared_gpu':True,'whole_epoch_speedup_not_inferred':True}
    write(root/'step_benchmark.json',result)
    print('STEP_BENCHMARK',json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    root=args.root.resolve()
    if (root/'run').exists():raise FileExistsError('Refusing to overwrite existing pilot run')
    for name in ['preflight','run']:(root/name).mkdir()
    torch.set_num_threads(4);torch.cuda.set_device(0);device=torch.device('cuda:0')
    existing.verify_runtime=verify_v1_runtime
    runtime=verify_v1_runtime()
    config=copy.deepcopy(existing.validate_protocol(existing.PROTOCOL,'unet2d',check_cache=True))
    config['training']['epochs']=3
    config['training']['total_optimizer_updates']=3*config['training']['updates_per_epoch']
    protocol={'kind':'independent_engineering_pilot_not_scientific_version3','epochs':3,
              'equations':'V2 unchanged: exp(delta*A)*h_old+B*T; D*T and z applied with original T',
              'implementation':'existing CUDA scan receives T/delta, delta, A, B, C; D=None,z=None inside kernel',
              'config':config,'test_checkpoint_selection':'validation loss only',
              'source_sha256':{n:sha256_file(PROJECT/n) for n in ['Models/TinyViM/mamba_bottleneck_v2_cuda_pilot.py','source/bottleneck_mamba/experiment_v2_cuda_pilot.py','source/bottleneck_mamba/test_v2_cuda_pilot.py']},
              'runtime_audit':runtime,'sharing':'GPU1; other tasks unchanged, shared load may affect timing'}
    write(root/'protocol.json',protocol)
    before=protection();write(root/'protected_before.json',before)
    write(root/'pilot_status.json',{'status':'preflight_and_benchmark','pid':os.getpid(),'epochs':3})
    api=existing.baseline_api()
    benchmark(config,api,root,device)
    api.seed_all(42);model=build().to(device)
    loaders,generator=make_fast_loaders(config,api,workers=2,validation_batch=8)
    criterion=api.BceDiceLoss().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.01,betas=(0.9,0.999),eps=1e-8)
    report=existing.preflight(model,loaders,criterion,optimizer,device,root/'preflight',[])
    write(root/'preflight/preflight_status.json',report)
    close(loaders);del model,optimizer,criterion;torch.cuda.empty_cache()
    # Preflight updates are discarded; reset both model initialization and loader plan.
    api.seed_all(42);model=build().to(device)
    loaders,generator=make_fast_loaders(config,api,workers=2,validation_batch=8)
    criterion=api.BceDiceLoss().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.01,betas=(0.9,0.999),eps=1e-8)
    provenance={'model':'unet2d_v2_cuda_equivalent_pilot_3epochs','seed':42,
                'protocol_sha256':sha256_file(root/'protocol.json'),'parameters':sum(p.numel() for p in model.parameters()),
                'split_sha256':config['data']['split_sha256'],'fresh_start':True,'physical_gpu':1,
                'engineering':{'isolated_pilot':True,'epochs':3,'normal_300_epoch_experiment':False}}
    write(root/'run/provenance.json',provenance)
    write(root/'pilot_status.json',{'status':'training','pid':os.getpid(),'epochs':3})
    existing.train(model,loaders,criterion,optimizer,device,root/'run',config,provenance,generator,resume=False)
    close(loaders);del loaders,model,optimizer;torch.cuda.empty_cache()
    # Rebuild on CPU and strictly load the validation-selected best before final test.
    cp=root/'run/checkpoints/best.pth';model=build()
    model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False),strict=True)
    checkpoint_hash=sha256_file(cp)
    val=json.loads((root/'run/validation/summary.json').read_text())
    data=Path(PATHS['active_split']).parent
    if sha256_file(data/'split.csv')!=provenance['split_sha256']:raise ValueError('Split changed')
    with (data/'split.csv').open() as f:ids={r['image_id'] for r in csv.DictReader(f) if r['split']=='test'}
    if len(ids)!=2020:raise ValueError('Unexpected test size')
    dataset=api.HAMDataset(data,'test')
    if dataset.training:raise ValueError('Test augmentation enabled')
    loader=DataLoader(dataset,batch_size=1,shuffle=False,num_workers=2,pin_memory=True,multiprocessing_context='spawn')
    model=model.to(device).eval()
    write(root/'pilot_status.json',{'status':'testing','pid':os.getpid(),'epochs':3,'images':2020})
    started=time.monotonic();summary=api.evaluate(model,loader,criterion,device,root/'test')
    summary.update(split='test',selected_epoch=val['selected_epoch'],checkpoint=str(cp),checkpoint_sha256=checkpoint_hash,
                   elapsed_seconds=time.monotonic()-started,engineering_pilot=True,training_epochs=3,
                   test_used_for_checkpoint_selection=False,not_comparable_to_300_epoch_accuracy=True)
    audit=validate_outputs(root/'test',ids,summary)
    if sha256_file(cp)!=checkpoint_hash:raise ValueError('Pilot best changed')
    write(root/'test/summary.json',summary);write(root/'test/artifact_audit.json',audit)
    after=protection();write(root/'protected_after.json',after)
    if before['sha256']!=after['sha256']:raise ValueError('Protected files changed')
    if before['active_training']['pid']!=after['active_training']['pid']:raise ValueError('Original training PID changed')
    with (root/'run/history.csv').open() as f:history=list(csv.DictReader(f))
    if [int(r['epoch']) for r in history]!=[1,2,3]:raise ValueError('Pilot did not complete all three epochs')
    result={'status':'complete','epochs':3,'history':history,'test':summary,'test_audit':audit,
            'benchmark':json.loads((root/'step_benchmark.json').read_text()),'protected_sources_and_completed_weights_unchanged':True,
            'original_training_before':before['active_training'],'original_training_after':after['active_training'],
            'output_root':str(root),'equations_changed':False,'scientific_version3_started':False}
    write(root/'pilot_result.json',result)
    write(root/'pilot_status.json',{'status':'complete','pid':os.getpid(),'epochs':3,'test_images':2020,'finished_at':datetime.now().astimezone().isoformat()})
    print('PILOT_COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        root=Path(sys.argv[sys.argv.index('--root')+1])
        if root.exists():write(root/'pilot_failure.json',{'error':repr(error),'pid':os.getpid(),'time':datetime.now().astimezone().isoformat()})
        raise

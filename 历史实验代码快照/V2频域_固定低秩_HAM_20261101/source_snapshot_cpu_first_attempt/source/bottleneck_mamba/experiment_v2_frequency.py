"""Extra independent arms: TinyViM+V2 and UNet2D+V1. GPU sharing authorized.

Reuse unchanged data/loss/training/evaluation; do not mutate active V1 sources.
"""
import argparse
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'source/comparison'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch import nn
import ham_fast_runner as existing
from ham_fast_pipeline import make_fast_loaders
from pretrained_unet2d import PretrainedUNet2D, prepare_imports, sha256_file
from experiment_v1 import verify_v1_runtime

ARMS = {
 'tinyvim_v2_freq_fixed': {'name':'tinyvim_v2_freq_fixed','base':'tinyvim_s','feature':['batch',512,8,8],'tokens':64},
 'tinyvim_v2_freq_dynamic': {'name':'tinyvim_v2_freq_dynamic','base':'tinyvim_s','feature':['batch',512,8,8],'tokens':64},
}

def build_model(arm, pretrained=True):
    if arm not in ARMS: raise ValueError(arm)
    prepare_imports('mambavision_t')
    from Models.TinyViM.mamba_bottleneck_v2_frequency import BottleneckMambaV2Frequency
    model=PretrainedUNet2D('tinyvim_s',pretrained=pretrained,new_layers_seed=42)
    model.align[-1]=nn.Sequential(model.align[-1],BottleneckMambaV2Frequency(
        channels=512,height=8,width=8,mode=arm.rsplit('_',1)[1],rank=4,
        epsilon=0.1,alpha_init=0.01,d_state=16,seed=44,frequency_seed=45))
    return model

def bottleneck_of(model,arm): return model.align[-1][1]

def source_hashes():
    paths=set(json.loads((PROJECT/'source/config/runtime_sources.json').read_text()))
    paths.update(['Models/TinyViM/mamba_bottleneck_v2.py',
        'Models/TinyViM/mamba_bottleneck_v2_frequency.py',
        'source/config/runtime_sources.json','source/config/protocol_v5_dedup_seed42.json',
        'source/bottleneck_mamba/experiment_extra.py','source/bottleneck_mamba/experiment_v1.py',
        'source/bottleneck_mamba/final_test_pipeline.py',
        'source/bottleneck_mamba/test_v2_frequency.py',
        'source/bottleneck_mamba/experiment_v2_frequency.py',
        'source/bottleneck_mamba/final_test_v2_frequency.py',
        'source/bottleneck_mamba/run_v2_frequency.sh','source/bottleneck_mamba/status_v2_frequency.sh'])
    return {p:sha256_file(PROJECT/p) for p in sorted(paths)}

def snapshot(root, hashes):
    import shutil
    for relative,digest in hashes.items():
        target=root/'source_snapshot'/relative
        if target.exists():
            if sha256_file(target)!=digest: raise ValueError('Snapshot changed: '+relative)
        else:
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(PROJECT/relative,target)
        if sha256_file(target)!=digest: raise ValueError('Snapshot copy mismatch')


from project_paths import current_protocol, original_source, original_repository_source, CACHE
import csv

def validate_protocol(protocol_path, model_name, check_cache=True):
    verify_v1_runtime()
    config = current_protocol(protocol_path)
    if config['version'] != 'ham_dedup_v5_seed42_20261023':
        raise ValueError('Wrong protocol version')
    if sha256_file(protocol_path) != protocol_path.with_suffix('.sha256').read_text().strip():
        raise ValueError('Protocol SHA256 mismatch')
    if model_name not in config['models']:
        raise ValueError('Model is outside the fixed three-encoder plus two-baseline scope')
    for relative, expected in config['source_lock'].items():
        if sha256_file(original_repository_source(relative)) != expected:
            raise ValueError('Frozen source changed: ' + relative)
    for absolute, expected in config['external_source_lock'].items():
        if sha256_file(original_source(absolute)) != expected:
            raise ValueError('Frozen official model source changed: ' + absolute)
    split = Path(config['data']['split_file'])
    if sha256_file(split) != config['data']['split_sha256']:
        raise ValueError('Revised split hash mismatch')
    if sha256_file(split.parent / 'split_audit.json') != config['data']['split_audit_sha256']:
        raise ValueError('Split audit hash mismatch')
    for filename, expected in config['data']['exclusion_artifact_hashes'].items():
        if sha256_file(split.parent / filename) != expected:
            raise ValueError('Exclusion decision/manifest changed: ' + filename)
    with split.open() as stream:
        rows = list(csv.DictReader(stream))
    counts = {s: sum(r['split'] == s for r in rows) for s in ('train', 'val', 'test')}
    if counts != config['data']['counts']:
        raise ValueError('Split counts differ from the protocol')
    if len({r['image_id'] for r in rows}) != len(rows):
        raise ValueError('Repeated image_id')
    active_ids = {r['image_id'] for r in rows}
    policy = json.loads((split.parent / 'exclusion_policy.json').read_text())
    for excluded, kept in policy['excluded_to_kept'].items():
        if excluded in active_ids or kept not in active_ids:
            raise ValueError('Approved photographic-variant exclusion not applied')
    for key in ('lesion_id', 'dedup_group_id'):
        sets = {s: {r[key] for r in rows if r['split'] == s} for s in counts}
        for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
            if sets[a] & sets[b]:
                raise ValueError(f'Cross-split {key}: {a}, {b}')
    if check_cache:
        for name, record in config['data']['cache_files'].items():
            path = CACHE / name
            if path.resolve() != Path(record['real_path']).resolve() or sha256_file(path) != record['sha256']:
                raise ValueError('Cache identity/content changed: ' + name)
    for name, record in config['models'].items():
        if record['initialization'] != 'random':
            source = record['pretrained_source']
            if sha256_file(source['weights_path']) != source['sha256']:
                raise ValueError('Pretrained weights changed: ' + name)
    if config['training']['epochs'] != 300 or config['training']['batch_size'] != 8:
        raise ValueError('Fixed training budget changed')
    if (counts['train'] + 7) // 8 != config['training']['updates_per_epoch']:
        raise ValueError('Incorrect update budget')
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=list(ARMS), required=True)
    parser.add_argument('--mode', choices=['cpu', 'preflight', 'train'], required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    arm = ARMS[args.arm]
    root = args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    runtime = verify_v1_runtime()
    config = copy.deepcopy(validate_protocol(existing.PROTOCOL, arm['base'], check_cache=True))
    existing.write_json(root/'runtime_validation_audit.json', runtime)
    config['models'][arm['name']] = {'initialization': 'ImageNet encoder and fresh decoder/bottleneck'
                                  if arm['base']=='tinyvim_s' else 'all random initialization',
                                  'run_dir': str(root/'run')}
    model_description = {
        'arm': args.arm, 'base': arm['base'], 'feature': arm['feature'],
        'token_shape': ['batch', arm['tokens'], 512], 'blocks': 1,
        'd_model': 512, 'd_state': 16, 'expand': 2, 'new_bottleneck_seed': 44,
        'scan': 'single forward row-major Mamba-1', 'outer': 'LayerNorm, mixer, residual',
        'all_parameters_trainable': True, 'v1_and_v2_combined': False,
    }
    model_description.update(
        formula='Y=V2(X)+alpha*irfft2(G*rfft2(X)); complete frequency branch, no X subtraction',
        frequency_mode=args.arm.rsplit('_',1)[1],rank=4,epsilon=0.1,alpha_init=0.01,
        frequency_seed=45,frequency_generator='GAP -> Linear(512,32) -> GELU -> separate U/V heads' if args.arm.endswith('dynamic') else 'fixed U[512,4], V[4,40]',
        initialization='normal std .02 for both factors or generator weights/biases; CPU proposal unchanged',
        fft='FP32 ortho, inverse s=(8,8), real gain conjugate boundary averaging',
        state_update='original V2 exp(delta*A)*h_old+B*U; original V2 wrapper directly called',
        sampler='unchanged make_fast_loaders reference RNG',training_scope='all parameters')
    protocol = {'version': args.arm, 'parent_protocol': str(existing.PROTOCOL),
                'parent_protocol_sha256': sha256_file(existing.PROTOCOL),
                'data': config['data'], 'training': config['training'],
                'preprocessing': config['preprocessing'], 'evaluation': config['evaluation'],
                'model': model_description, 'source_sha256': source_hashes(),
                'test_data_accessed': False, 'gpu_sharing_authorized': True,
                'gpu_utilization_not_an_idle_gate': True}
    protocol_path = root/'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise RuntimeError('Protocol differs from saved run')
    else:
        existing.write_json(protocol_path, protocol)
    snapshot(root,protocol['source_sha256'])
    existing.write_json(root/'config.json',config)
    stage = root/({'cpu':'cpu_check','preflight':'preflight','train':'run'}[args.mode])
    stage.mkdir(exist_ok=True)
    if args.mode=='train':
        passed = json.loads((root/'preflight/preflight_status.json').read_text())
        if passed['status']!='passed' or passed['protocol_sha256']!=sha256_file(protocol_path):
            raise RuntimeError('Same-protocol preflight required')
        if any((stage/n).exists() for n in ['checkpoints','status.json','history.csv']):
            raise FileExistsError('Refusing to overwrite existing training')
    lock = root/'RUNNING.lock'
    with lock.open('x') as f:
        f.write(str(os.getpid()))
    loaders = {}
    try:
        signal.signal(signal.SIGUSR1, existing.request_stop)
        api = existing.baseline_api()
        api.seed_all(42)
        torch.set_num_threads(4)
        if args.mode=='cpu':
            device=torch.device('cpu')
        else:
            torch.cuda.set_device(args.gpu)
            device=torch.device(f'cuda:{args.gpu}')
        physical_gpu = int(os.environ.get('EXPERIMENT_PHYSICAL_GPU', args.gpu))
        print(f'MODE={args.mode} ARM={args.arm} PID={os.getpid()} PHYSICAL_GPU={physical_gpu} CUDA_INDEX={args.gpu}', flush=True)
        model = build_model(args.arm, pretrained=True).to(device)
        module = bottleneck_of(model,args.arm)
        observed=[]
        def observe(m,inputs,output):
            if not observed:
                record={'input':list(inputs[0].shape),'output':list(output.shape)}
                existing.write_json(stage/'bottleneck_shape.json',record)
                print('REAL_BOTTLENECK_SHAPE',json.dumps(record),flush=True)
            observed.append(True)
        if args.mode=='preflight':
            module.register_forward_hook(observe)
        if args.mode=='cpu':
            from experiment_extra import build_model as original_build
            api.seed_all(42)
            reference=original_build('tinyvim_v2',pretrained=True)
            reference_rng=torch.get_rng_state().clone()
            api.seed_all(42)
            model=build_model(args.arm,pretrained=True)
            module=bottleneck_of(model,args.arm)
            assert torch.equal(reference_rng,torch.get_rng_state()), 'CPU initialization RNG drift'
            original=reference.state_dict(); candidate=model.state_dict()
            for key,value in original.items():
                mapped=key.replace('align.3.1.','align.3.1.v2.')
                assert torch.equal(value,candidate[mapped]), key
            del reference,original,candidate
            shape=[]
            module.register_forward_hook(lambda m,i,o:shape.append(list(i[0].shape)))
            model.train()
            x=torch.rand(2,3,256,256)
            y=(torch.rand(2,1,256,256)>.5).float()
            loss=api.BceDiceLoss()(model(x),y);loss.backward()
            bad=[n for n,p in model.named_parameters() if not p.requires_grad or p.grad is None or not torch.isfinite(p.grad).all()]
            assert not bad,bad
            assert shape==[[2,512,8,8]],shape
            assert not torch.cuda.is_initialized()
            report={'passed':True,'shape':shape,'loss':float(loss),'all_parameters_finite_gradients':True,
                'old_v2_and_nonfrequency_weights_bitwise_equal':True,'initialization_rng_equal':True,
                'device':'cpu','cuda_initialized':False,'protocol_sha256':sha256_file(protocol_path)}
            existing.write_json(stage/'checks.json',report)
            print('FULL_MODEL_CPU_PASSED',json.dumps(report),flush=True)
            return
        cpu=json.loads((root/'cpu_check/checks.json').read_text())
        if not cpu['passed'] or cpu['protocol_sha256']!=sha256_file(protocol_path): raise ValueError('Same protocol CPU check required')
        import subprocess
        external=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],text=True)
        existing.write_json(stage/'gpu_external_before.json',{'compute':external,'sharing_authorized':True})
        print('EXTERNAL_COMPUTE_BEFORE',external,flush=True)
        loaders,generator=make_fast_loaders(config,api,workers=2,validation_batch=8)
        criterion=api.BceDiceLoss().to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.01,
                                   betas=(0.9,0.999),eps=1e-8)
        info={'model':arm['name'],'seed':42,'mode':args.mode,'pid':os.getpid(),
              'protocol_file':str(protocol_path),'protocol_sha256':sha256_file(protocol_path),
              'split_sha256':config['data']['split_sha256'],'physical_gpu':physical_gpu,
              'process_cuda_index':args.gpu,
              'gpu_name':torch.cuda.get_device_name(device),
              'parameters':sum(p.numel() for p in model.parameters()),
              'new_bottleneck_parameters':sum(p.numel() for p in module.parameters()),
              'initialization':config['models'][arm['name']]['initialization'],
              'training':config['training'],'source_sha256':source_hashes(),
              'fresh_start':True,'test_data_accessed':False,'gpu_sharing_authorized':True,
              'engineering':{'runner':'unchanged ham_fast_runner.train/preflight','workers':2,'validation_batch':8}}
        existing.write_json(stage/'provenance.json',info)
        if hasattr(model,'encoder'):
            existing.write_json(stage/'weight_loading_audit.json',model.encoder.loading_audit)
        print('MODEL_PARAMETERS',info['parameters'],'NEW_BOTTLENECK',info['new_bottleneck_parameters'],flush=True)
        if args.mode=='preflight':
            report=existing.preflight(model,loaders,criterion,optimizer,device,stage,[])
            if not observed:raise AssertionError('Bottleneck did not execute')
            assert json.loads((stage/'bottleneck_shape.json').read_text())['input']==[8,512,8,8]
            report.update(all_parameter_gradients_finite=True,optimizer='AdamW lr=.001 wd=.01',protocol_sha256=info['protocol_sha256'],split_sha256=config['data']['split_sha256'])
            existing.write_json(stage/'preflight_status.json',report)
            print('PREFLIGHT_PASSED',json.dumps(report),flush=True)
        else:
            existing.train(model,loaders,criterion,optimizer,device,stage,config,info,generator,resume=False)
    except BaseException as error:
        existing.write_json(stage/'failure.json',{'error':repr(error),'pid':os.getpid(),
                            'mode':args.mode,'time':datetime.now().isoformat()})
        raise
    finally:
        for loader in loaders.values():
            if hasattr(loader,'close'):loader.close()
        lock.unlink(missing_ok=True)


if __name__=='__main__':
    main()

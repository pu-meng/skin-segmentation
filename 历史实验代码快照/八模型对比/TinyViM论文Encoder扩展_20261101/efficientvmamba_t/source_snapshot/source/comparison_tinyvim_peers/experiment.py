"""Official TinyViM comparison encoders; identical V5 data, decoder, optimizer and budget."""
import argparse
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'source/comparison'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(PROJECT / 'source/bottleneck_mamba'))

import torch
from torch import nn
import ham_fast_runner as existing
from ham_fast_pipeline import make_fast_loaders
from pretrained_unet2d import PretrainedUNet2D, prepare_imports, sha256_file
from experiment_v1 import verify_v1_runtime

RUNS_ROOT = PROJECT/'log/八模型对比/TinyViM论文Encoder扩展_20261101'
ARMS = {
    'swiftformer_s': {'name':'swiftformer_s_unet', 'base':'tinyvim_s', 'channels':[48,64,168,224]},
    'efficientformerv2_s0': {'name':'efficientformerv2_s0_unet', 'base':'tinyvim_s', 'channels':[32,48,96,176]},
    'mobileone_s1': {'name':'mobileone_s1_unet', 'base':'tinyvim_s', 'channels':[96,192,512,1280]},
    'efficientvmamba_t': {'name':'efficientvmamba_t_unet', 'base':'tinyvim_s', 'channels':[48,96,192,384]},
}


def build_model(arm, pretrained=True):
    from Models.TinyViMPeers.model import build_model as make
    return make(arm, pretrained=pretrained)


def source_hashes():
    names = ['Models/TinyViMPeers/model.py', 'Models/TinyViMPeers/__init__.py',
             'source/comparison_tinyvim_peers/experiment.py',
             'source/comparison_tinyvim_peers/final_test.py',
             'source/comparison_tinyvim_peers/launch.py',
             'source/bottleneck_mamba/experiment_v1.py', 'source/bottleneck_mamba/final_test_pipeline.py',
             'source/comparison/ham_fast_runner.py', 'source/comparison/ham_fast_pipeline.py',
             'source/comparison/pretrained_unet2d.py', 'source/comparison/unet2d.py']
    for repo in ('SwiftFormer','EfficientFormer','MobileOne','EfficientVMamba'):
        root=PROJECT/'references/code'/repo
        names.extend(str(f.relative_to(PROJECT)) for f in root.rglob('*')
                     if f.is_file() and '.git' not in f.parts and f.suffix in ('.py','.yaml','.yml'))
    return {name:sha256_file(PROJECT/name) for name in sorted(set(names))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=list(ARMS), required=True)
    parser.add_argument('--mode', choices=['prepare', 'preflight', 'train'], required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    arm = ARMS[args.arm]
    root = args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    runtime = verify_v1_runtime()
    existing.verify_runtime = verify_v1_runtime
    config = copy.deepcopy(existing.validate_protocol(existing.PROTOCOL, arm['base'], check_cache=True))
    existing.write_json(root/'runtime_validation_audit.json', runtime)
    config['models'][arm['name']] = {'initialization': 'official ImageNet encoder; fresh shared decoder and alignment',
                                  'run_dir': str(root/'run')}
    model_description = {
        'arm':args.arm, 'name':arm['name'], 'initialization':'official ImageNet weights',
        'replacement':'encoder only; no extra bottleneck module',
        'encoder_channels':arm['channels'], 'feature_spatial_sizes':[64,32,16,8],
        'decoder':'unchanged SharedDecoder', 'alignment_channels':[64,128,256,512],
        'decoder_seed':42, 'alignment_seed':43, 'all_parameters_trainable':True,
        'input':'RGB /255 then fixed ImageNet mean/std, same as TinyViM comparison',
        'parent_validation_template':'tinyvim_s; only shared data/training configuration reused',
    }
    protocol = {'version': args.arm, 'parent_protocol': str(existing.PROTOCOL),
                'parent_protocol_sha256': sha256_file(existing.PROTOCOL),
                'data': config['data'], 'training': config['training'],
                'preprocessing': config['preprocessing'], 'evaluation': config['evaluation'],
                'model': model_description, 'source_sha256': source_hashes(),
                'official_material': json.loads((PROJECT/'references/核对记录/tinyvim_peers_20261101'/(args.arm+'.json')).read_text()),
                'test_data_accessed': False, 'gpu_sharing_authorized': True,
                'gpu_utilization_not_an_idle_gate': True}
    protocol_path = root/'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise RuntimeError('Protocol differs from saved run')
    else:
        existing.write_json(protocol_path, protocol)
    snapshot = root/'source_snapshot'
    for relative, expected in protocol['source_sha256'].items():
        destination = snapshot/relative
        if destination.exists():
            if sha256_file(destination) != expected:
                raise RuntimeError('Saved source snapshot differs: '+relative)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT/relative, destination)
    if args.mode == 'prepare':
        print('PROTOCOL_AND_SOURCE_SNAPSHOT_READY', protocol_path, flush=True)
        return
    stage = root/('preflight' if args.mode=='preflight' else 'run')
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
        torch.cuda.set_device(args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
        physical_gpu = int(os.environ.get('EXPERIMENT_PHYSICAL_GPU', args.gpu))
        print(f'MODE={args.mode} ARM={args.arm} PID={os.getpid()} PHYSICAL_GPU={physical_gpu} CUDA_INDEX={args.gpu}', flush=True)
        model = build_model(args.arm, pretrained=True).to(device)
        module = model.encoder
        observed=[]
        def observe(m,inputs,output):
            if not observed:
                record={'input':list(inputs[0].shape),'features':[list(value.shape) for value in output]}
                existing.write_json(stage/'encoder_shapes.json',record)
                print('REAL_ENCODER_SHAPES',json.dumps(record),flush=True)
            observed.append(True)
        if args.mode=='preflight':
            module.register_forward_hook(observe)
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
              'encoder_parameters':sum(p.numel() for p in module.parameters()),
              'initialization':config['models'][arm['name']]['initialization'],
              'training':config['training'],'source_sha256':source_hashes(),
              'fresh_start':True,'test_data_accessed':False,'gpu_sharing_authorized':True,
              'engineering':{'runner':'unchanged ham_fast_runner.train/preflight','workers':2,'validation_batch':8}}
        existing.write_json(stage/'provenance.json',info)
        if hasattr(model,'encoder'):
            existing.write_json(stage/'weight_loading_audit.json',model.encoder.loading_audit)
        print('MODEL_PARAMETERS',info['parameters'],'ENCODER_PARAMETERS',info['encoder_parameters'],flush=True)
        if args.mode=='preflight':
            report=existing.preflight(model,loaders,criterion,optimizer,device,stage,[])
            if not observed:raise AssertionError('Encoder did not execute')
            report.update(protocol_sha256=info['protocol_sha256'],split_sha256=config['data']['split_sha256'])
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

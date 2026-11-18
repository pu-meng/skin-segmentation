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
    'tinyvim_v2': {'name': 'tinyvim_s_bottleneck_mamba_v2', 'base': 'tinyvim_s',
                   'feature': ['batch', 512, 8, 8], 'tokens': 64},
    'unet_v1': {'name': 'unet2d_bottleneck_mamba_v1', 'base': 'unet2d',
                'feature': ['batch', 512, 16, 16], 'tokens': 256},
}


def build_model(arm, pretrained=True):
    prepare_imports('mambavision_t')
    if arm == 'tinyvim_v2':
        from Models.TinyViM.mamba_bottleneck_v2 import BottleneckMambaV2
        model = PretrainedUNet2D('tinyvim_s', pretrained=pretrained, new_layers_seed=42)
        model.align[-1] = nn.Sequential(model.align[-1], BottleneckMambaV2(512, 16, seed=44))
    elif arm == 'unet_v1':
        from unet2d import UNet2D
        from Models.TinyViM.mamba_bottleneck_v1 import BottleneckMambaV1
        model = UNet2D()
        # Original fifth encoder output is [b,512,16,16] for RGB256.
        # Insert after its convolutions and before the first decoder upsample.
        model.encoders[-1] = nn.Sequential(model.encoders[-1], BottleneckMambaV1(512, 16, seed=44))
    else:
        raise ValueError(arm)
    return model


def bottleneck_of(model, arm):
    return model.align[-1][1] if arm == 'tinyvim_v2' else model.encoders[-1][1]


def source_hashes():
    files = ['Models/TinyViM/mamba_bottleneck_v1.py', 'Models/TinyViM/mamba_bottleneck_v2.py',
             'source/bottleneck_mamba/experiment_extra.py', 'source/bottleneck_mamba/experiment_v1.py',
             'source/bottleneck_mamba/test_extra.py', 'source/comparison/ham_fast_runner.py',
             'source/comparison/ham_fast_pipeline.py', 'source/comparison/pretrained_unet2d.py',
             'source/comparison/unet2d.py']
    return {p: sha256_file(PROJECT/p) for p in files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=list(ARMS), required=True)
    parser.add_argument('--mode', choices=['preflight', 'train'], required=True)
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
    if args.arm == 'tinyvim_v2':
        model_description.update(generator='original x_proj split rho/B/C, dt_proj and softplus unchanged',
                                 state_update='exp(delta*A)*h_old + B*U',
                                 scan_implementation='explicit float32 PyTorch recurrence; no CUDA package modification',
                                 D_skip_and_z_gate='unchanged')
    else:
        model_description.update(generator='U -> causal DWConv3 -> SiLU -> PWConv(E,E+2N), direct split',
                                 delta='softplus of first E outputs; no rho or dt_proj',
                                 state_update='exp(delta*A)*h_old + delta*B*U',
                                 scan_implementation='installed selective_scan CUDA/autograd',
                                 input_normalization='original UNet RGB/255; no added ImageNet normalization')
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

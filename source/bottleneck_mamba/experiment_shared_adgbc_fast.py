"""Accelerated Shared AD-GBC fresh restart, original segmentation-only protocol."""
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

import torch
from torch import nn
import ham_fast_runner as existing
from ham_fast_pipeline import make_fast_loaders
from pretrained_unet2d import PretrainedUNet2D, prepare_imports, sha256_file
from experiment_v1 import verify_v1_runtime

ARM = 'tinyvim_base_shared_adgbc_s0_fast'
ARMS = {ARM: {'name': ARM, 'base': 'tinyvim_s'}}


def build_model(arm, pretrained=True):
    if arm != ARM:
        raise ValueError(arm)
    from Models.SharedADGBC.fast_model import FastTinyViMSharedADGBC
    return FastTinyViMSharedADGBC(pretrained=pretrained)


def adapter_of(model, arm):
    return model.shared_adgbc


def source_hashes():
    files = ['Models/SharedADGBC/fast_model.py', 'Models/SharedADGBC/fast_distance.py', 'Models/SharedADGBC/model.py', 'Models/SharedADGBC/__init__.py',
             'Models/SharedADGBC/official_granular_ball.py',
             'references/code/AD-GBC/archs_GBC.py',
             'source/bottleneck_mamba/experiment_shared_adgbc_fast.py',
             'source/bottleneck_mamba/test_shared_adgbc.py',
             'source/bottleneck_mamba/final_test_shared_adgbc_fast.py',
             'source/bottleneck_mamba/run_shared_adgbc_fast.py',
             'source/bottleneck_mamba/experiment_v1.py',
             'source/bottleneck_mamba/final_test_pipeline.py',
             'source/comparison/ham_fast_runner.py', 'source/comparison/ham_fast_pipeline.py',
             'source/comparison/pretrained_unet2d.py', 'source/comparison/unet2d.py']
    return {p: sha256_file(PROJECT/p) for p in files}


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
    acceptance=PROJECT/'log/研究记录_辅助产物/V2与SharedADGBC加速验收/report.json'
    accepted=json.loads(acceptance.read_text())
    assert accepted['status']=='passed'
    for relative in ('Models/SharedADGBC/fast_model.py','Models/SharedADGBC/fast_distance.py'):
        assert sha256_file(PROJECT/relative)==accepted['fast_source_sha256'][relative],relative
    runtime = verify_v1_runtime()
    existing.verify_runtime = verify_v1_runtime
    config = copy.deepcopy(existing.validate_protocol(existing.PROTOCOL, arm['base'], check_cache=True))
    existing.write_json(root/'runtime_validation_audit.json', runtime)
    config['models'][arm['name']] = {'initialization': 'ImageNet encoder and fresh paired baseline decoder/alignment/shared AD-GBC'
                                  if arm['base']=='tinyvim_s' else 'all random initialization',
                                  'run_dir': str(root/'run')}
    model_description = {
        'arm':args.arm,'base':'tinyvim_s','parent_version':'original TinyViM-S + UNet baseline',
        'location':'encoder aligned S0 and decoder D2 upsampled feature, before original channel concat',
        'input_shapes':[[64,64,64],[64,64,64]],'shared_module_instances':1,'calls_per_forward':2,
        'call_order':'encoder-side S0 first, decoder-side D2 upsample second',
        'encoder_deeper_path':'original encoder forward; refine only decoder-bound S0 copy',
        'fusion':'concat(shared(D2_up),shared(S0)) -> original DoubleConv128to64',
        'shared_components':'centers, log_sigma, refine Conv3x3/BN/ReLU, including BN running statistics',
        'module':'FastGranularBall subclass of pinned official GranularBall; same parameters and formula',
        'backend':'FP32 ATen forward chunks of 512 positions; fused Triton backward; auxiliary dif omitted',
        'restart':'fresh ImageNet encoder and seed42/43/44 layers; no prior HAM weights or optimizer loaded',
        'supersedes_run':str(PROJECT/'log/TinyViM_基础模型_Shared_ADGBC_最浅Skip_HAM_20261102'),
        'source_commit':'8786893f75ee5777a692fd742f1f03d9fb92e6f0',
        'num_balls':32,'feature_dimension':64,'use_diag_cov':True,'use_residual':True,'tau':1.,
        'projection':'none: in_channels equals proj_dim64',
        'scope':'shared-mechanism transfer, not full paper reproduction; public code broadcasts learned centers directly',
        'geometry_loss_weights':{'diversity':0.,'scale_consistency':0.},
        'loss':'unchanged baseline BCE+Dice; first controlled shared-module ablation',
        'added_parameters':41088,'parameters':8716429,'all_parameters_trainable':True,
        'HAM_checkpoint_loaded':False,'decoder_seed':42,'alignment_seed':43,'shared_module_seed':44,
        'bottleneck':'baseline only; no V2,V4,LoRA or other skip refiner',
        'initial_identity_claimed':False,
    }
    protocol = {'version': args.arm, 'parent_protocol': str(existing.PROTOCOL),
                'parent_protocol_sha256': sha256_file(existing.PROTOCOL),
                'data': config['data'], 'training': config['training'],
                'preprocessing': config['preprocessing'], 'evaluation': config['evaluation'],
                'model': model_description, 'source_sha256': source_hashes(),
                'numerical_acceptance_report':str(acceptance),
                'numerical_acceptance_report_sha256':sha256_file(acceptance),
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
        module = adapter_of(model,args.arm)
        routing_audit = None
        if args.mode == 'preflight':
            model.eval()
            with torch.no_grad():
                probe=torch.rand(1,3,256,256,generator=torch.Generator().manual_seed(42044)).to(device)
                features=[layer(f) for layer,f in zip(model.align,model.raw_features(probe))]
                expected=model.decoder(features)
                actual=model.decode_features(features,refine=False)
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                routing_audit={'bypass_matches_baseline_decoder_exactly':True,
                               'encoder_S0_decoder_D2_before_concat':True,'max_abs_bypass_error':float((actual-expected).abs().max())}
            existing.write_json(stage/'routing_audit.json',routing_audit)
        observed=[]
        def observe(m,inputs,output):
            if not observed:
                record={'input':list(inputs[0].shape),'output':list(output[0].shape),'shared_module_id':id(m)}
                existing.write_json(stage/'shared_shape.json',record)
                print('REAL_SHARED_ADGBC_SHAPE',json.dumps(record),flush=True)
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
              'new_skip_parameters':sum(p.numel() for p in module.parameters()),
              'initialization':config['models'][arm['name']]['initialization'],
              'training':config['training'],'source_sha256':source_hashes(),
              'fresh_start':True,'test_data_accessed':False,'gpu_sharing_authorized':True,
              'engineering':{'runner':'unchanged ham_fast_runner.train/preflight','workers':2,'validation_batch':8,
                             'backend':'ATen chunks512 forward + Triton backward','fresh_accelerated_restart':True}}
        existing.write_json(stage/'provenance.json',info)
        if hasattr(model,'encoder'):
            existing.write_json(stage/'weight_loading_audit.json',model.encoder.loading_audit)
        print('MODEL_PARAMETERS',info['parameters'],'NEW_SKIP',info['new_skip_parameters'],flush=True)
        if args.mode=='preflight':
            before=[p.detach().cpu().clone() for p in module.parameters()]
            report=existing.preflight(model,loaders,criterion,optimizer,device,stage,[])
            assert all(not torch.equal(a,p.detach().cpu()) for a,p in zip(before,module.parameters()))
            assert all(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters())
            assert info['parameters']==8716429 and info['new_skip_parameters']==41088
            report.update(routing_audit,added_parameters=41088,num_balls=32,shared_call_count=len(observed),
                          added_parameter_tensors_updated=len(before),all_shared_parameter_gradients_nonzero=True,
                          test_loader_constructed=False)
            if len(observed)<6:raise AssertionError('Expected two shared calls per training batch')
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

"""Four independent baseline S0 refiners; unchanged HAM training math."""
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

RUN = PROJECT/'log/TinyViM_基础模型_最浅Skip四组_HAM_20261102'
ARMS = {
    'spatial': {'name':'tinyvim_base_shallow_spatial','base':'tinyvim_s','label':'基础模型＋最浅Skip空间门控','added':4129},
    'channel': {'name':'tinyvim_base_shallow_channel','base':'tinyvim_s','label':'基础模型＋最浅Skip通道门控','added':1096},
    'conv': {'name':'tinyvim_base_shallow_conv','base':'tinyvim_s','label':'基础模型＋最浅Skip卷积变换','added':4672},
    'region': {'name':'tinyvim_base_shallow_region','base':'tinyvim_s','label':'基础模型＋最浅Skip区域重构','added':512},
}


def build_model(arm, pretrained=True):
    if arm not in ARMS:
        raise ValueError(arm)
    from Models.ShallowSkipRefiners.model import TinyViMShallowSkipRefiner
    return TinyViMShallowSkipRefiner(arm,pretrained=pretrained)


def adapter_of(model, arm):
    return model.skip_refiner


def source_hashes():
    files = ['Models/ShallowSkipRefiners/model.py', 'Models/ShallowSkipRefiners/__init__.py',
             'source/bottleneck_mamba/experiment_shallow_skip_refiners.py',
             'source/bottleneck_mamba/test_shallow_skip_refiners.py',
             'source/bottleneck_mamba/final_test_shallow_skip_refiners.py',
             'source/bottleneck_mamba/run_shallow_skip_refiners.py',
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
    runtime = verify_v1_runtime()
    existing.verify_runtime = verify_v1_runtime
    config = copy.deepcopy(existing.validate_protocol(existing.PROTOCOL, arm['base'], check_cache=True))
    existing.write_json(root/'runtime_validation_audit.json', runtime)
    config['models'][arm['name']] = {'initialization': 'ImageNet encoder and fresh paired baseline decoder/alignment/refiner'
                                  if arm['base']=='tinyvim_s' else 'all random initialization',
                                  'run_dir': str(root/'run')}
    descriptions = {
        'spatial': {'operator':'sigmoid(score(ReLU(Ws*S0 + Wu*D2_up)))*S0',
                    'hidden_channels':32,'projections_bias':False,'score_bias':True,
                    'decoder_conditioned':True,'reference':'Attention U-Net inspired additive gate; simplified, no BN'},
        'channel': {'operator':'sigmoid(up(ReLU(down(global_average(S0)))))*S0',
                    'hidden_channels':8,'bias':True,'decoder_conditioned':False,
                    'reference':'SE-style channel gate; global pooling, bottleneck ratio8'},
        'conv': {'operator':'PW1x1(DW3x3(S0))','stride':1,'padding':1,'bias':False,
                 'normalization':None,'activation':None,'residual':False,'decoder_conditioned':False},
        'region': {'operator':'cosine assignments -> per-image weighted region means -> membership broadcast',
                   'regions':8,'temperature':.1,'normalization':'L2 across feature channels for assignments',
                   'assignment_axis':'softmax over8 regions','aggregation_axis':'normalize over4096 inputs',
                   'epsilon':1e-6,'residual':False,'decoder_conditioned':False,
                   'reference':'simple one-pass aggregate/broadcast baseline; not AD-GBC or Slot Attention reproduction'},
    }
    model_description = {
        'arm':args.arm,'base':'tinyvim_s','parent_version':'original TinyViM-S + UNet baseline',
        'bottleneck':'baseline only; no V2, V4, LoRA or frequency module',
        'location':'aligned S0, after48to64 projection and before decoder D2 channel concat',
        'skip_shape':['batch',64,64,64], 'D2_context_shape':['batch',64,64,64],
        'unchanged_connections':'S1->D1; S2->D0; deepest512x8x8 main path',
        'adapter':descriptions[args.arm], 'added_parameters':arm['added'],
        'parameters':8675341+arm['added'],'all_parameters_trainable':True,
        'HAM_checkpoint_loaded':False,'decoder_seed':42,'alignment_seed':43,'refiner_seed':44,
        'initialization':'official ImageNet encoder and fresh paired baseline decoder/alignment/refiner',
        'initial_identity_claimed':False,
    }
    protocol = {'version': arm['name'], 'parent_protocol': str(existing.PROTOCOL),
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
                routing_audit={'refiner_bypass_matches_baseline_decoder_exactly':True,
                               'changed_only_S0_before_D2':True,'skip_shape':list(features[0].shape),
                               'max_abs_bypass_error':float((actual-expected).abs().max())}
            existing.write_json(stage/'routing_audit.json',routing_audit)
        observed=[]
        def observe(m,inputs,output):
            if not observed:
                record={'inputs':[list(x.shape) for x in inputs],'output':list(output.shape)}
                existing.write_json(stage/'refiner_shape.json',record)
                print('REAL_S0_D2_SHAPE',json.dumps(record),flush=True)
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
              'engineering':{'runner':'unchanged ham_fast_runner.train/preflight','workers':2,'validation_batch':8}}
        existing.write_json(stage/'provenance.json',info)
        if hasattr(model,'encoder'):
            existing.write_json(stage/'weight_loading_audit.json',model.encoder.loading_audit)
        print('MODEL_PARAMETERS',info['parameters'],'NEW_SKIP',info['new_skip_parameters'],flush=True)
        if args.mode=='preflight':
            before=[p.detach().cpu().clone() for p in module.parameters()]
            report=existing.preflight(model,loaders,criterion,optimizer,device,stage,[])
            assert all(not torch.equal(a,p.detach().cpu()) for a,p in zip(before,module.parameters()))
            assert all(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters())
            assert info['parameters']==8675341+arm['added'] and info['new_skip_parameters']==arm['added']
            report.update(routing_audit,added_parameters=arm['added'],
                          added_parameter_tensors_updated=len(before),all_refiner_gradients_nonzero=True,
                          test_loader_constructed=False)
            if not observed:raise AssertionError('Shallow skip refiner did not execute')
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

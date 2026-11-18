"""One independent HAM baseline: four encoder scales, four decoder stages."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import signal
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'source/comparison'), str(Path(__file__).parent)]
import torch
import ham_fast_runner as runner
from ham_fast_pipeline import make_fast_loaders
from pretrained_unet2d import sha256_file
from experiment_v1 import verify_v1_runtime
from Models.TinyViM.four_stage_decoder import build_four_stage_baseline

NAME = 'tinyvim_four_stage_decoder'
LABEL = 'TinyViM-S＋四层对应Decoder基础版'
DEFAULT_ROOT = ROOT/'log/TinyViM_四层对应Decoder_基础_HAM_20261104'


def source_hashes():
    paths = [
        'Models/TinyViM/four_stage_decoder.py', 'Models/TinyViM/model.py',
        'source/bottleneck_mamba/experiment_four_stage_decoder.py',
        'source/bottleneck_mamba/final_test_four_stage_decoder.py',
        'source/bottleneck_mamba/run_four_stage_decoder.sh',
        'source/bottleneck_mamba/experiment_v1.py',
        'source/bottleneck_mamba/final_test_pipeline.py',
        'source/comparison/ham_fast_runner.py', 'source/comparison/ham_fast_pipeline.py',
        'source/comparison/pretrained_unet2d.py', 'source/comparison/unet2d.py',
        'source/comparison/baseline_runner_seed42.py', 'source/comparison/project_paths.py',
    ]
    return {p: sha256_file(ROOT/p) for p in paths}


def prepare(root):
    runner.verify_runtime = verify_v1_runtime
    runtime = verify_v1_runtime()
    config = copy.deepcopy(runner.validate_protocol(runner.PROTOCOL, 'tinyvim_s', check_cache=True))
    config['models'][NAME] = dict(
        initialization='official ImageNet encoder; fresh four-stage decoder42; align43',
        run_dir=str(root/'run'))
    protocol = dict(
        version=NAME, parent_protocol=str(runner.PROTOCOL),
        parent_protocol_sha256=sha256_file(runner.PROTOCOL),
        data=config['data'], training=config['training'], preprocessing=config['preprocessing'],
        evaluation=config['evaluation'], source_sha256=source_hashes(), test_data_accessed=False,
        model=dict(encoder='unchanged official pretrained TinyViM-S', encoder_stages=4,
                   alignment=[64,128,256,512], decoder_stages=4, decoder_order=['D3','D2','D1','D0'],
                   decoder_output_shapes=[[512,8,8],[256,16,16],[128,32,32],[64,64,64]],
                   deepest='DoubleConv512to512; no duplicated self-skip',
                   skips=['S2 -> D2','S1 -> D1','S0 -> D0'],
                   output_head='Conv1x1 64to1 -> bilinear logits to256 align_corners=False -> sigmoid',
                   note='D3 is a new trainable deepest stage; entire decoder and head change together',
                   initialization_seeds=[42,43], all_parameters_trainable=True,
                   laplacian=False, bottleneck_v2=False))
    pp = root/'protocol.json'
    if pp.exists() and json.loads(pp.read_text()) != protocol:
        raise RuntimeError('Saved protocol/source identity changed; refusing overwrite')
    runner.write_json(pp, protocol)
    runner.write_json(root/'runtime_validation_audit.json', runtime)
    for relative, digest in protocol['source_sha256'].items():
        target = root/'source_snapshot'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha256_file(target) != digest:
            raise RuntimeError('Source snapshot changed: ' + relative)
        if not target.exists():
            shutil.copyfile(ROOT/relative, target)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['prepare','preflight','train'], required=True)
    parser.add_argument('--run-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    root = args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    stage = root/('run' if args.mode == 'train' else 'preflight')
    stage.mkdir(exist_ok=True)
    lock = root/'RUNNING.lock'
    fd = lock.open('x')
    fd.write(str(os.getpid())); fd.close()
    loaders = {}
    try:
        config = prepare(root)
        if args.mode == 'prepare':
            print('PROTOCOL_PREPARED', root/'protocol.json', flush=True)
            return
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; GPU preflight/training has not started')
        if args.mode == 'train':
            report = json.loads((root/'preflight/preflight_status.json').read_text())
            if report.get('status') != 'passed' or report['protocol_sha256'] != sha256_file(root/'protocol.json'):
                raise RuntimeError('Same-protocol real-data preflight required')
            if any((stage/n).exists() for n in ('history.csv','status.json','checkpoints')):
                raise FileExistsError('Existing training; refusing overwrite')
        signal.signal(signal.SIGUSR1, runner.request_stop)
        api = runner.baseline_api(); api.seed_all(42)
        torch.set_num_threads(4); torch.cuda.set_device(args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
        model = build_four_stage_baseline(pretrained=True).to(device)
        loaders, generator = make_fast_loaders(config, api, workers=2, validation_batch=8)
        criterion = api.BceDiceLoss().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01,
                                     betas=(.9,.999), eps=1e-8)
        info = dict(model=NAME, seed=42, mode=args.mode, pid=os.getpid(),
                    protocol_file=str(root/'protocol.json'), protocol_sha256=sha256_file(root/'protocol.json'),
                    split_sha256=config['data']['split_sha256'], source_sha256=source_hashes(),
                    parameters=sum(p.numel() for p in model.parameters()),
                    initialization=config['models'][NAME]['initialization'],
                    physical_gpu=int(os.environ.get('EXPERIMENT_PHYSICAL_GPU',args.gpu)),
                    process_cuda_index=args.gpu, gpu_name=torch.cuda.get_device_name(device),
                    training=config['training'], fresh_start=True, test_data_accessed=False)
        runner.write_json(stage/'provenance.json',info)
        runner.write_json(stage/'weight_loading_audit.json',model.encoder.loading_audit)
        if args.mode == 'preflight':
            report = runner.preflight(model,loaders,criterion,optimizer,device,stage,[])
            report.update(protocol_sha256=sha256_file(root/'protocol.json'),
                          split_sha256=config['data']['split_sha256'])
            runner.write_json(stage/'preflight_status.json',report)
            print('PREFLIGHT_PASSED',json.dumps(report),flush=True)
        else:
            runner.train(model,loaders,criterion,optimizer,device,stage,config,info,generator,resume=False)
    except BaseException as error:
        runner.write_json(stage/'failure.json',dict(error=repr(error),pid=os.getpid(),mode=args.mode))
        raise
    finally:
        for loader in loaders.values():
            if hasattr(loader,'close'): loader.close()
        lock.unlink()


if __name__ == '__main__':
    main()

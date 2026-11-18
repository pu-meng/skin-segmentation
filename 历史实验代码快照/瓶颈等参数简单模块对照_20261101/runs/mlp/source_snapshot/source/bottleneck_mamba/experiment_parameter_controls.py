"""HAM-only, fixed-protocol TinyViM MLP/Conv controls and final-test handoff."""
import argparse
import copy
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'source/comparison'), str(Path(__file__).parent)]
import torch
from torch.utils.data import DataLoader
import ham_fast_runner as parent
from ham_fast_pipeline import make_fast_loaders
from pretrained_unet2d import sha256_file
from experiment_v1 import verify_v1_runtime
from parameter_control_models import build
from final_test_pipeline import fixed_best_record, training_ready, validate_outputs, ProgressLoader

LOG = ROOT/'log/瓶颈等参数简单模块对照_20261101'
EXPECTED = {'mlp': 10371202, 'conv': 10372805}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    parent.write_json(path, value)


def sources():
    paths = list((ROOT/'Models/TinyViM').glob('*.py'))
    paths += [ROOT/'source/bottleneck_mamba'/n for n in (
        'parameter_control_models.py', 'experiment_parameter_controls.py',
        'check_parameter_controls.py', 'launch_parameter_controls.py',
        'final_test_pipeline.py', 'experiment_v1.py')]
    paths += [ROOT/'source/comparison'/n for n in (
        'ham_fast_runner.py', 'ham_fast_pipeline.py', 'baseline_runner_seed42.py',
        'pretrained_unet2d.py', 'project_paths.py')]
    paths += list((ROOT/'references/code/TinyViM').rglob('*.py'))
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(set(paths))}


def configuration(kind, work):
    parent.verify_runtime = verify_v1_runtime
    audit = verify_v1_runtime()
    config = copy.deepcopy(parent.validate_protocol(parent.PROTOCOL, 'tinyvim_s', check_cache=True))
    name = 'tinyvim_parameter_control_' + kind
    config['models'][name] = dict(initialization='ImageNet encoder; fresh original decoder/alignment; seed44 new block', run_dir=str(work/'run'))
    protocol = dict(version=name+'_ham_seed42_20261101',
        parent_protocol=str(parent.PROTOCOL), parent_protocol_sha256=sha256_file(parent.PROTOCOL),
        data=config['data'], training=config['training'], preprocessing=config['preprocessing'],
        evaluation=config['evaluation'], original_model=config['models']['tinyvim_s'],
        model=dict(name=name, kind=kind, parameters=EXPECTED[kind], target_added_parameters=1695744,
                   feature=['batch',512,8,8], seed=42, bottleneck_seed=44,
                   insertion='last alignment output, before first decoder',
                   mlp_width=1653 if kind=='mlp' else None,
                   conv_width=184 if kind=='conv' else None,
                   common_wrapper='LayerNorm then mixer then residual',
                   base_training_and_initialization_unchanged=True),
        source_sha256=sources(), test_never_selects_weights=True)
    p = work/'protocol.json'
    if p.exists():
        if json.loads(p.read_text()) != protocol:
            raise RuntimeError('Saved protocol/source identity changed; refusing run')
    else:
        write(p,protocol)
    for relative, digest in protocol['source_sha256'].items():
        destination = work/'source_snapshot'/relative
        if destination.exists():
            assert sha256_file(destination)==digest,relative
        else:
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/relative,destination)
    write(work/'runtime_validation_audit.json',audit)
    return config,name,protocol


def final_test(kind, work, config, name, protocol):
    run = work/'run'
    assert training_ready(run), 'Full training and validation required'
    fixed = fixed_best_record(run)
    assert fixed['history_rows']==300
    output = work/'test'
    assert not output.exists(), 'Refusing overwrite of test outputs'
    assert not torch.cuda.is_initialized(), 'Strict CPU loading must precede CUDA initialization'
    model = build('tinyvim',kind,pretrained=False)
    model.load_state_dict(torch.load(fixed['checkpoint'],map_location='cpu',weights_only=False),strict=True)
    write(work/'test_cpu_load_audit.json',dict(strict_load_passed=True,**fixed))
    assert not torch.cuda.is_initialized()
    split = Path(config['data']['split_file'])
    assert sha256_file(split)==fixed['split_sha256']
    members = list(csv.DictReader(split.open()))
    ids = {r['image_id'] for r in members if r['split']=='test'}
    assert len(ids)==2020
    assert not ids & {r['image_id'] for r in members if r['split']!='test'}
    api = parent.baseline_api()
    dataset = api.HAMDataset(split.parent,'test')
    assert not dataset.training and len(dataset)==2020
    loader = DataLoader(dataset,batch_size=1,shuffle=False,num_workers=2,pin_memory=True,multiprocessing_context='spawn')
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    model = model.to(device).eval()
    criterion = api.BceDiceLoss().to(device)
    state = dict(status='testing',model=name,pid=os.getpid(),total_images=2020,**fixed)
    write(work/'test_manifest.json',dict(**fixed,protocol_sha256=sha256_file(work/'protocol.json'),
        source_sha256=protocol['source_sha256'],dataset='HAM10000',test_augmentation=False,tta=False,threshold=.5))
    summary = api.evaluate(model,ProgressLoader(loader,work/'test_status.json',state),criterion,device,output)
    summary.update(dataset='HAM10000',split='test',model=name,**fixed)
    assert sha256_file(fixed['checkpoint'])==fixed['checkpoint_sha256']
    write(output/'summary.json',summary)
    m=summary['per_image_mean']
    (output/'report.txt').write_text(f'HAM10000 TinyViM＋{kind}；300轮；验证损失best第{fixed["selected_epoch"]}轮\n2020张独立测试；逐图Dice {100*m["dice"]:.4f}%；IoU {100*m["iou"]:.4f}%；HD95 {m["hd95_pixels"]:.4f}像素\n')
    write(work/'test_artifact_audit.json',validate_outputs(output,ids,summary))
    write(work/'test_status.json',dict(state,status='test_complete',completed_images=2020))
    print('TEST_COMPLETE',name,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--kind',choices=list(EXPECTED),required=True)
    parser.add_argument('--mode',choices=['prepare','preflight','train','test'],required=True)
    args=parser.parse_args()
    work=LOG/'runs'/args.kind
    work.mkdir(parents=True,exist_ok=True)
    with (work/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config,name,protocol=configuration(args.kind,work)
        if args.mode=='prepare':
            print('PROTOCOL_READY',work/'protocol.json',flush=True)
            return
        torch.set_num_threads(4)
        if args.mode=='test':
            final_test(args.kind,work,config,name,protocol)
            return
        stage=work/('preflight' if args.mode=='preflight' else 'run')
        stage.mkdir(exist_ok=True)
        if args.mode=='train':
            passed=json.loads((work/'preflight/preflight_status.json').read_text())
            assert passed['status']=='passed' and passed['protocol_sha256']==sha256_file(work/'protocol.json')
            assert not any((stage/n).exists() for n in ['checkpoints','history.csv','status.json']), 'Refusing overwrite'
        loaders={}
        try:
            api=parent.baseline_api()
            api.seed_all(42)
            torch.cuda.set_device(0)
            device=torch.device('cuda:0')
            model=build('tinyvim',args.kind,pretrained=True).to(device)
            count=sum(p.numel() for p in model.parameters())
            assert count==EXPECTED[args.kind]
            block=model.align[-1][1]
            observed=[]
            if args.mode=='preflight':
                def shape_hook(module,inputs,output):
                    assert list(inputs[0].shape[1:])==[512,8,8]
                    assert output.shape==inputs[0].shape
                    observed.append(list(output.shape))
                block.register_forward_hook(shape_hook)
            loaders,generator=make_fast_loaders(config,api,workers=2,validation_batch=8)
            criterion=api.BceDiceLoss().to(device)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8)
            info=dict(model=name,seed=42,pid=os.getpid(),parameters=count,
                new_bottleneck_parameters=block.parameter_count,protocol_file=str(work/'protocol.json'),
                protocol_sha256=sha256_file(work/'protocol.json'),split_sha256=config['data']['split_sha256'],
                physical_gpu=int(os.environ.get('EXPERIMENT_PHYSICAL_GPU','0')),
                source_sha256=protocol['source_sha256'],training=config['training'],
                fresh_start=True,test_data_accessed=False,
                initialization=config['models'][name]['initialization'],
                engineering=dict(runner='unchanged ham_fast_runner',workers=2,validation_batch=8))
            write(stage/'provenance.json',info)
            write(stage/'weight_loading_audit.json',model.encoder.loading_audit)
            signal.signal(signal.SIGUSR1,parent.request_stop)
            if args.mode=='preflight':
                before=[p.detach().cpu().clone() for p in block.parameters()]
                report=parent.preflight(model,loaders,criterion,optimizer,device,stage,[])
                assert observed[0]==[8,512,8,8]
                changed=sum(not torch.equal(a,p.detach().cpu()) for a,p in zip(before,block.parameters()))
                assert changed==len(before), 'Not all added parameter tensors updated'
                report.update(protocol_sha256=info['protocol_sha256'],parameters=count,
                    added_parameter_tensors_updated=changed,bottleneck_shape=observed[0],
                    test_loader_constructed=False)
                write(stage/'preflight_status.json',report)
                print('PREFLIGHT_PASSED',args.kind,json.dumps(report),flush=True)
            else:
                parent.train(model,loaders,criterion,optimizer,device,stage,config,info,generator,resume=False)
                assert training_ready(stage)
        except BaseException as error:
            write(stage/'failure.json',dict(error=repr(error),mode=args.mode,pid=os.getpid(),time=datetime.now().astimezone().isoformat()))
            raise
        finally:
            for loader in loaders.values():
                if hasattr(loader,'close'):
                    loader.close()
                elif getattr(loader,'_iterator',None) is not None:
                    loader._iterator._shutdown_workers()


if __name__=='__main__':
    main()

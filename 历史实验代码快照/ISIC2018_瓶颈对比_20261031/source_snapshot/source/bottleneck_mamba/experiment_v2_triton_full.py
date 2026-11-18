"""Independent formal UNet2D+V2 Triton run: 300 epochs -> validation best -> test.

Fresh seed42 initialization; never resume or overwrite the stopped Python run
or 3-epoch pilot. Uses the already verified no-division Triton implementation.
"""
from pathlib import Path
from datetime import datetime
import argparse
import copy
import csv
import json
import os
import sys
import time
PROJECT=Path(__file__).resolve().parents[2]
MODEL_NAME='unet2d_bottleneck_mamba_v2_triton_full'


def make_formal_config(parent,root):
    t=parent['training']
    if t['epochs']!=300 or t['batch_size']!=8 or t['total_optimizer_updates']!=300*t['updates_per_epoch']:
        raise ValueError('Formal run requires unchanged 300-epoch batch8 parent budget')
    config=copy.deepcopy(parent)
    config['models'][MODEL_NAME]={'initialization':'random','run_dir':str(root/'run')}
    return config


def build():
    from experiment_unet_v2 import build_model
    from Models.TinyViM.mamba_bottleneck_v2_triton_pilot import BottleneckMambaV2TritonPilot
    model=build_model('unet_v2')
    original=model.encoders[-1][1]
    replacement=BottleneckMambaV2TritonPilot(512,16,seed=44)
    replacement.load_state_dict(original.state_dict(),strict=True)
    model.encoders[-1][1]=replacement
    return model


def close(loaders):
    for value in loaders.values():
        if hasattr(value,'close'):value.close()


def main():
    sys.path[:0]=[str(PROJECT),str(PROJECT/'source/comparison'),str(Path(__file__).parent)]
    import torch
    from torch.utils.data import DataLoader
    import ham_fast_runner as existing
    from ham_fast_pipeline import make_fast_loaders
    from experiment_v1 import verify_v1_runtime
    from experiment_v2_triton_pilot import protection
    from pretrained_unet2d import sha256_file
    from final_test_pipeline import validate_outputs,ProgressLoader
    from project_paths import PATHS
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    root=args.root.resolve()
    if (root/'run').exists():raise FileExistsError('Refusing to overwrite any prior formal run')
    (root/'preflight').mkdir();(root/'run').mkdir()
    write=existing.write_json
    torch.set_num_threads(4);torch.cuda.set_device(0);device=torch.device('cuda:0')
    existing.verify_runtime=verify_v1_runtime
    runtime=verify_v1_runtime()
    config=make_formal_config(existing.validate_protocol(existing.PROTOCOL,'unet2d',check_cache=True),root)
    before=protection();write(root/'protected_before.json',before)
    sources=['Models/TinyViM/mamba_bottleneck_v2_triton_pilot.py','Models/TinyViM/mamba_bottleneck_v2.py',
             'source/bottleneck_mamba/experiment_v2_triton_full.py','source/bottleneck_mamba/test_v2_triton_full.py',
             'source/bottleneck_mamba/test_v2_triton_pilot.py','source/bottleneck_mamba/experiment_v2_triton_pilot.py',
             'source/comparison/ham_fast_runner.py','source/comparison/ham_fast_pipeline.py','source/comparison/unet2d.py']
    protocol={'kind':'formal_300_epoch_UNet2D_V2_Triton','epochs':300,'seed':42,'bottleneck_seed':44,
              'fresh_start':True,'resume_from_stopped_run':False,'resume_from_3_epoch_pilot':False,
              'equations':'exp(delta*A)*H_old+B*T; original rho/B/C, D*T, z gate and residual retained',
              'backend':'verified direct FP32 Triton forward/backward, no delta division; shared-gradient atomic reductions',
              'all_parameters_trainable':True,'physical_gpu':1,'config':config,'runtime_audit':runtime,
              'source_sha256':{n:sha256_file(PROJECT/n) for n in sources},
              'completion_contract':{'epochs':300,'updates':config['training']['total_optimizer_updates'],
                                     'selection':'minimum validation loss','final_test_images':2020,
                                     'predictions_and_csv':True,'test_visualizations':12,'test_never_selects_checkpoint':True}}
    write(root/'protocol.json',protocol)
    api=existing.baseline_api()
    status={'status':'preflight','pid':os.getpid(),'epochs':300,'physical_gpu':1,'started_at':datetime.now().astimezone().isoformat()}
    write(root/'formal_status.json',status)
    api.seed_all(42);model=build().to(device)
    loaders,generator=make_fast_loaders(config,api,workers=2,validation_batch=8)
    criterion=api.BceDiceLoss().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.01,betas=(0.9,0.999),eps=1e-8)
    preflight=existing.preflight(model,loaders,criterion,optimizer,device,root/'preflight',[])
    write(root/'preflight/preflight_status.json',preflight)
    print('FORMAL_PREFLIGHT_PASSED',json.dumps(preflight),flush=True)
    close(loaders);del model,optimizer,criterion;torch.cuda.empty_cache()
    api.seed_all(42);model=build().to(device)
    loaders,generator=make_fast_loaders(config,api,workers=2,validation_batch=8)
    criterion=api.BceDiceLoss().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.01,betas=(0.9,0.999),eps=1e-8)
    info={'model':MODEL_NAME,'seed':42,'parameters':sum(p.numel() for p in model.parameters()),
          'protocol_sha256':sha256_file(root/'protocol.json'),'split_sha256':config['data']['split_sha256'],
          'physical_gpu':1,'fresh_start':True,'source_sha256':protocol['source_sha256'],
          'engineering':{'backend':'direct Triton V2','formal_epochs':300,'pilot':False}}
    write(root/'run/provenance.json',info)
    status.update(status='training');write(root/'formal_status.json',status)
    print('FORMAL_TRAINING_START',json.dumps(info),flush=True)
    existing.train(model,loaders,criterion,optimizer,device,root/'run',config,info,generator,resume=False)
    close(loaders);del loaders,model,optimizer;torch.cuda.empty_cache()
    with (root/'run/history.csv').open() as f:history=list(csv.DictReader(f))
    if [int(x['epoch']) for x in history]!=list(range(1,301)):raise ValueError('Incomplete formal training history')
    val=json.loads((root/'run/validation/summary.json').read_text())
    winner=min(history,key=lambda x:float(x['val_loss']))
    if int(winner['epoch'])!=val['selected_epoch']:raise ValueError('Validation best selection mismatch')
    cp=root/'run/checkpoints/best.pth';digest=sha256_file(cp)
    if digest!=val['checkpoint_sha256']:raise ValueError('Best identity mismatch')
    # Strictly reconstruct/load on CPU before beginning final test inference.
    model=build();model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False),strict=True)
    write(root/'test_cpu_load_audit.json',{'strict':True,'map_location':'cpu','checkpoint_sha256':digest})
    data=Path(PATHS['active_split']).parent
    if sha256_file(data/'split.csv')!=config['data']['split_sha256']:raise ValueError('Split changed')
    with (data/'split.csv').open() as f:ids={x['image_id'] for x in csv.DictReader(f) if x['split']=='test'}
    if len(ids)!=2020:raise ValueError('Unexpected test size')
    dataset=api.HAMDataset(data,'test')
    if dataset.training:raise ValueError('Unexpected test augmentation')
    loader=DataLoader(dataset,batch_size=1,shuffle=False,num_workers=2,pin_memory=True,multiprocessing_context='spawn')
    model=model.to(device).eval()
    status.update(status='testing',selected_epoch=val['selected_epoch'],total_images=2020,completed_images=0)
    write(root/'formal_status.json',status)
    start=time.monotonic()
    result=api.evaluate(model,ProgressLoader(loader,root/'formal_status.json',status),criterion,device,root/'test')
    result.update(split='test',model=MODEL_NAME,seed=42,training_epochs=300,selected_epoch=val['selected_epoch'],
                  checkpoint=str(cp),checkpoint_sha256=digest,split_sha256=config['data']['split_sha256'],
                  test_used_for_checkpoint_selection=False,parameters=info['parameters'],elapsed_seconds=time.monotonic()-start)
    audit=validate_outputs(root/'test',ids,result)
    if sha256_file(cp)!=digest:raise ValueError('Best changed during test')
    write(root/'test/summary.json',result);write(root/'test/artifact_audit.json',audit)
    after=protection();write(root/'protected_after.json',after)
    if before['sha256']!=after['sha256']:raise ValueError('Protected original files changed')
    if protocol['source_sha256']!={n:sha256_file(PROJECT/n) for n in sources}:raise ValueError('Formal sources changed')
    status.update(status='training_validation_test_complete',completed_images=2020,finished_at=datetime.now().astimezone().isoformat())
    write(root/'formal_status.json',status)
    train_state=json.loads((root/'run/status.json').read_text());train_state['status']='training_validation_test_complete';write(root/'run/status.json',train_state)
    (root/'test/report.txt').write_text(f'UNet2D+V2 Triton正式300轮，验证选第{val["selected_epoch"]}轮best。测试2020张，整体Dice {100*result["global"]["dice"]:.4f}%，逐图平均Dice {100*result["per_image_mean"]["dice"]:.4f}%。测试不参与选权重。')
    print('FORMAL_PIPELINE_COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        root=Path(sys.argv[sys.argv.index('--root')+1])
        if root.exists():
            (root/'formal_failure.json').write_text(json.dumps({'error':repr(error),'pid':os.getpid(),'time':datetime.now().astimezone().isoformat()},indent=2))
        raise

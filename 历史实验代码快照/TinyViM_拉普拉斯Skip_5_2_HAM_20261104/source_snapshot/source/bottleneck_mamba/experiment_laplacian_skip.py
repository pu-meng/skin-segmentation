"""Single authorized section-5.2 experiment; reuse frozen HAM training math."""
import argparse,copy,json,os,signal,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(Path(__file__).parent)]
import torch
import ham_fast_runner as runner
from ham_fast_pipeline import make_fast_loaders
from pretrained_unet2d import sha256_file
from experiment_v1 import verify_v1_runtime
NAME='tinyvim_laplacian_skip_52'
DEFAULT_ROOT=ROOT/'log/TinyViM_拉普拉斯Skip_5_2_HAM_20261104'
LABEL='V0＋拉普拉斯Skip 5.2'

def build_model(pretrained=True):
    from Models.LaplacianSkip.model import TinyViMLaplacianSkip
    return TinyViMLaplacianSkip(pretrained=pretrained)

def source_hashes():
    paths=['Models/LaplacianSkip/model.py','Models/LaplacianSkip/__init__.py',
           'Models/TinyViM/blocks.py','source/bottleneck_mamba/experiment_laplacian_skip.py',
           'source/bottleneck_mamba/final_test_laplacian_skip.py',
           'source/bottleneck_mamba/run_laplacian_skip.sh',
           'source/comparison/ham_fast_runner.py','source/comparison/ham_fast_pipeline.py',
           'source/comparison/pretrained_unet2d.py','source/comparison/unet2d.py',
           'source/bottleneck_mamba/experiment_v1.py','source/bottleneck_mamba/final_test_pipeline.py']
    return {p:sha256_file(ROOT/p) for p in paths}

def contract(model,device):
    model.eval()
    with torch.no_grad():
        image=torch.randn(1,3,256,256,device=device)
        features=[a(f) for a,f in zip(model.align,model.raw_features(image))]
        base=model.decoder(features)
        torch.testing.assert_close(base,model.decode_features(features,refine=False),rtol=0,atol=0)
        identity=model.decode_features(features,identity=True)
        torch.testing.assert_close(base,identity,rtol=1e-5,atol=1e-6)
        s=features[0];d=torch.randn_like(s)
        _,_,parts=model.skip_refiner(s,d,return_details=True)
        torch.testing.assert_close(parts['expanded']+parts['h'],s,rtol=1e-5,atol=1e-6)
        for key in ('gate_h','gate_l','gate_d'):
            torch.testing.assert_close(parts[key],torch.ones_like(parts[key]),rtol=0,atol=0)
        # RepDW structural reparameterization uses eval-mode frozen BN statistics.
        rep=model.skip_refiner.detail[0]
        torch.testing.assert_close(rep(s),rep.fuse()(s),rtol=1e-4,atol=1e-5)
    return dict(bypass_matches_baseline=True,identity_matches_baseline=True,
                decomposition_reconstructs=True,all_initial_gates_equal_one=True,
                repdw_eval_fusion_matches=True,scan_tokens=1024,scan_directions=2,
                gate_shapes={k:list(parts[k].shape) for k in ('gate_h','gate_l','gate_d')})

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['preflight','train'],required=True)
    p.add_argument('--run-root',type=Path,default=DEFAULT_ROOT);p.add_argument('--gpu',type=int,default=0)
    a=p.parse_args();root=a.run_root.resolve();root.mkdir(parents=True,exist_ok=True)
    runtime=verify_v1_runtime();runner.verify_runtime=verify_v1_runtime
    config=copy.deepcopy(runner.validate_protocol(runner.PROTOCOL,'tinyvim_s',check_cache=True))
    config['models'][NAME]={'initialization':'official ImageNet encoder; paired fresh decoder42/align43; refiner44', 'run_dir':str(root/'run')}
    protocol=dict(version=NAME,parent_protocol=str(runner.PROTOCOL),parent_protocol_sha256=sha256_file(runner.PROTOCOL),
        data=config['data'],training=config['training'],preprocessing=config['preprocessing'],evaluation=config['evaluation'],
        model=dict(base='TinyViM-S + original UNet V0',location='S0 before D2 concat; BOTH skip and D2-up context gated',
                   channels=64,gaussian=[1,4,6,4,1],gaussian_normalizer_1d=16,padding='replicate',
                   reduction='Gaussian5 then ::2',expansion='bilinear align_corners=False then Gaussian5',
                   mamba=dict(d_model=64,d_state=16,d_conv=4,expand=2,directions='forward/reverse row-major; independent parameters; mean then PW1'),
                   detail='TinyViM RepDW -> GELU -> PW1; residual gamma=0.001/channel',
                   gates='3 separate PW128to32-GELU-Conv3to1; 2sigmoid; last conv zero-init; shared over channels',
                   context_scale=.001,initialization_seeds=[42,43,44],WPO=False,all_parameters_trainable=True),
        source_sha256=source_hashes(),test_data_accessed=False)
    pp=root/'protocol.json'
    if pp.exists() and json.loads(pp.read_text())!=protocol:raise RuntimeError('Protocol changed')
    runner.write_json(pp,protocol);runner.write_json(root/'runtime_validation_audit.json',runtime)
    for path in protocol['source_sha256']:
        dest=root/'source_snapshot'/path;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists() and sha256_file(dest)!=protocol['source_sha256'][path]:raise RuntimeError('Snapshot changed')
        if not dest.exists():shutil.copyfile(ROOT/path,dest)
    stage=root/('preflight' if a.mode=='preflight' else 'run');stage.mkdir(exist_ok=True)
    if a.mode=='train':
        report=json.loads((root/'preflight/preflight_status.json').read_text())
        assert report['status']=='passed' and report['protocol_sha256']==sha256_file(pp)
        if any((stage/x).exists() for x in ['status.json','history.csv','checkpoints']):raise FileExistsError('Existing training')
    lock=root/'RUNNING.lock';lock.open('x').write(str(os.getpid()));loaders={}
    try:
        signal.signal(signal.SIGUSR1,runner.request_stop)
        api=runner.baseline_api();api.seed_all(42);torch.set_num_threads(4);torch.cuda.set_device(a.gpu)
        device=torch.device(f'cuda:{a.gpu}');model=build_model().to(device)
        if a.mode=='preflight':runner.write_json(stage/'contract.json',contract(model,device))
        # Contract checks are eval-only; formal training always constructs fresh weights.
        loaders,generator=make_fast_loaders(config,api,workers=2,validation_batch=8)
        criterion=api.BceDiceLoss().to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8)
        info=dict(model=NAME,seed=42,mode=a.mode,pid=os.getpid(),protocol_file=str(pp),protocol_sha256=sha256_file(pp),
                  split_sha256=config['data']['split_sha256'],parameters=sum(p.numel() for p in model.parameters()),
                  new_skip_parameters=sum(p.numel() for p in model.skip_refiner.parameters()),
                  initialization=config['models'][NAME]['initialization'],physical_gpu=int(os.environ.get('EXPERIMENT_PHYSICAL_GPU',a.gpu)),
                  process_cuda_index=a.gpu,gpu_name=torch.cuda.get_device_name(device),training=config['training'],
                  source_sha256=source_hashes(),fresh_start=True,test_data_accessed=False)
        runner.write_json(stage/'provenance.json',info);runner.write_json(stage/'weight_loading_audit.json',model.encoder.loading_audit)
        print('PARAMETERS',info['parameters'],'NEW_SKIP',info['new_skip_parameters'],flush=True)
        if a.mode=='preflight':
            report=runner.preflight(model,loaders,criterion,optimizer,device,stage,[])
            grads={n:float(p.grad.abs().sum()) for n,p in model.skip_refiner.named_parameters()}
            if not all(v>0 for v in grads.values()):raise RuntimeError('Inactive refiner parameter after 3 updates: '+str(grads))
            report.update(protocol_sha256=sha256_file(pp),split_sha256=config['data']['split_sha256'],all_refiner_gradients_nonzero=True)
            runner.write_json(stage/'refiner_gradient_sums.json',grads);runner.write_json(stage/'preflight_status.json',report)
            print('PREFLIGHT_PASSED',json.dumps(report),flush=True)
        else:runner.train(model,loaders,criterion,optimizer,device,stage,config,info,generator,resume=False)
    except BaseException as error:
        runner.write_json(stage/'failure.json',dict(error=repr(error),pid=os.getpid(),mode=a.mode));raise
    finally:
        for loader in loaders.values():
            if hasattr(loader,'close'):loader.close()
        lock.unlink(missing_ok=True)
if __name__=='__main__':main()

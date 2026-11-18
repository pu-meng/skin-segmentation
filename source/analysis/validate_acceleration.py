"""Independent equivalence checks and paired speed probes; no formal run modified."""
import argparse
import copy
import gc
import hashlib
import io
import json
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison')]
import torch
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t')
from Models.SharedADGBC.fast_distance import direct_distance,quadratic_distance
from Models.SharedADGBC.official_granular_ball import GranularBall
from Models.SharedADGBC.fast_model import FastGranularBall,FastTinyViMSharedADGBC
from Models.SharedADGBC.model import TinyViMSharedADGBC
from Models.SkipLoRA.model import TinyViMV2SkipLoRA
from Models.SkipLoRA.fast_model import FastTinyViMV2SkipLoRA
from Models.TinyViM.mamba_bottleneck_v2 import scan_without_delta_injection
from Models.SkipLoRA.triton_scan import scan_v2_triton
from ham_fast_pipeline import PlannedDataset
from project_paths import PATHS
from ham_fast_runner import baseline_api


def error(a,b):
    a,b=a.detach().double(),b.detach().double()
    return {'max_abs':float((a-b).abs().max()),'relative_l2':float((a-b).norm()/a.norm().clamp_min(1e-12))}


def primitive(values,old,new,rtol=3e-4,atol=1e-3):
    left=[v.detach().clone().requires_grad_() for v in values]
    right=[v.detach().clone().requires_grad_() for v in values]
    a,b=old(*left),new(*right)
    torch.testing.assert_close(a,b,rtol=5e-5,atol=1e-4)
    grad=torch.randn_like(a)/a.numel()**.5
    ga=torch.autograd.grad((a*grad).sum(),left)
    gb=torch.autograd.grad((b*grad).sum(),right)
    for x,y in zip(ga,gb):
        assert torch.isfinite(y).all()
        torch.testing.assert_close(x,y,rtol=rtol,atol=atol)
    return {'output':error(a,b),'gradients':[error(x,y) for x,y in zip(ga,gb)]}


def paired_times(old,new,x,y,criterion,repeats=5):
    models={'old':old,'fast':new}
    opts={k:torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01) for k,m in models.items()}
    def step(key):
        m=models[key];m.train();opts[key].zero_grad(set_to_none=True)
        loss=criterion(m(x),y);loss.backward();opts[key].step()
    for key in models:
        for _ in range(2):step(key)
    times={k:[] for k in models};peaks={}
    for rep in range(repeats):
        for key in (('old','fast') if rep%2==0 else ('fast','old')):
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            initial=torch.cuda.memory_allocated();start=time.perf_counter()
            step(key);torch.cuda.synchronize()
            times[key].append((time.perf_counter()-start)*1000)
            peaks[key]=max(peaks.get(key,0),(torch.cuda.max_memory_allocated()-initial)/2**20)
    return {'milliseconds':times,'mean_ms':{k:statistics.mean(v) for k,v in times.items()},
        'median_ms':{k:statistics.median(v) for k,v in times.items()},'incremental_peak_mib':peaks,
        'mean_speedup':statistics.mean(times['old'])/statistics.mean(times['fast'])}


def network_check(old,new,state,x,y,criterion,train_mode=False,optimizer_state=None):
    old.load_state_dict(state,strict=True);new.load_state_dict(state,strict=True)
    assert old.state_dict().keys()==new.state_dict().keys()
    old.cuda().train(train_mode);new.cuda().train(train_mode)
    old.zero_grad(set_to_none=True);new.zero_grad(set_to_none=True)
    a,b=old(x),new(x)
    # Train-mode BN may amplify small FP32 reduction-order differences. Keep a
    # maximum absolute bound AND a tight whole-output norm bound; record mask flips.
    # Training includes BN and ReLU; isolated kink-adjacent pixels may amplify
    # ulp-level distance errors. Bound BOTH probability error and global L2.
    output_atol=2e-3 if train_mode else 5e-5
    torch.testing.assert_close(a,b,rtol=5e-4,atol=output_atol)
    assert error(a,b)['relative_l2']<1e-4,error(a,b)
    la,lb=criterion(a,y),criterion(b,y)
    la.backward();lb.backward()
    differences=[];grad_diff_sq=grad_ref_sq=0.
    for (na,pa),(nb,pb) in zip(old.named_parameters(),new.named_parameters()):
        assert na==nb and pa.grad is not None and pb.grad is not None
        assert torch.isfinite(pb.grad).all(),na
        torch.testing.assert_close(pa.grad,pb.grad,rtol=5e-3,atol=5e-4,msg=lambda msg:na+': '+msg)
        differences.append((na,error(pa.grad,pb.grad)))
        grad_diff_sq+=float((pa.grad.double()-pb.grad.double()).square().sum())
        grad_ref_sq+=float(pa.grad.double().square().sum())
    grad_rel=(grad_diff_sq/max(grad_ref_sq,1e-24))**.5
    assert grad_rel<1e-3,grad_rel
    gradient_worst=sorted(differences,key=lambda v:v[1]['max_abs'],reverse=True)[:5]
    # Check a whole AdamW update without allowing a large overall parameter drift.
    for model in (old,new):
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        if optimizer_state is not None:
            optimizer.load_state_dict(copy.deepcopy(optimizer_state))
        optimizer.step()
    sq_error=sq_ref=update_ref_sq=0.;max_abs=0.
    for (na,pa),(nb,pb) in zip(old.named_parameters(),new.named_parameters()):
        e=error(pa,pb);max_abs=max(max_abs,e['max_abs'])
        sq_error+=float((pa.detach().double()-pb.detach().double()).square().sum())
        sq_ref+=float(pa.detach().double().square().sum())
        update_ref_sq+=float((pa.detach().double()-state[na].to(pa.device).double()).square().sum())
    update_rel=(sq_error/max(sq_ref,1e-24))**.5
    update_step_rel=(sq_error/max(update_ref_sq,1e-24))**.5
    assert update_rel<2e-4,update_rel
    assert update_step_rel<.02,update_step_rel
    old.eval();new.eval()
    stream=io.BytesIO();torch.save(new.state_dict(),stream);stream.seek(0)
    with torch.no_grad():before=new(x)
    new.load_state_dict(torch.load(stream,map_location='cuda',weights_only=True),strict=True)
    with torch.no_grad():after=new(x)
    torch.testing.assert_close(before,after,rtol=0,atol=0)
    return {'mode':'train' if train_mode else 'eval','batch':x.shape[0],
        'output':error(a,b),'loss_abs':float((la-lb).abs()),'gradient_worst5':gradient_worst,
        'optimizer_state':'checkpoint' if optimizer_state is not None else 'fresh',
        'output_tolerance':{'rtol':5e-4,'atol':output_atol,'relative_l2_max':1e-4},
        'binary_prediction_changed_pixels':int(((a>=.5)!=(b>=.5)).sum()),
        'all_gradient_relative_l2':grad_rel,'all_gradient_relative_l2_max':1e-3,
        'gradient_tolerance':{'rtol':5e-3,'atol':5e-4},
        'adam_update_relative_l2':update_rel,'adam_update_max_abs':max_abs,
        'adam_update_step_relative_l2':update_step_rel,'adam_update_step_relative_l2_max':.02,
        'all_parameter_gradients_checked':len(differences),'strict_checkpoint_compatible':True,
        'fast_save_reload_exact':True}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',type=int,default=0)
    args=parser.parse_args()
    out=ROOT/'log/研究记录_辅助产物/V2与SharedADGBC加速验收'
    out.mkdir(parents=True,exist_ok=True)
    report={'status':'running','original_training_code_changed':False,'test_data_accessed':False,
            'scope':'FP32; same batches/checkpoint; GPU0 shared with other jobs; short paired probes, not isolated benchmarks',
            'numerical_limitations':[
                'Reordered FP32 reductions are not bitwise equivalent and do not guarantee identical training trajectories.',
                'An earlier prototype on a mature checkpoint with RESET AdamW had 0.0301 relative first-update error (2% check failed); trained-state continuation must retain its optimizer history.'
            ]}
    def save():
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    protected={}
    for name in ['TinyViM_V2_最浅Skip_LoRA_r8_20261101','TinyViM_基础模型_Shared_ADGBC_最浅Skip_HAM_20261102']:
        protected.update(json.loads((ROOT/'log'/name/'protocol.json').read_text())['source_sha256'])
    try:
        torch.set_num_threads(4);torch.manual_seed(42);torch.cuda.set_device(args.gpu)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
        report['distance_checks']=[];save()
        for B,N,K,D in [(2,19,5,7),(1,64,32,64),(8,4096,32,64)]:
            z=torch.randn(B,N,D,device='cuda')*.5;c=torch.randn(K,D,device='cuda')*.2
            s=(torch.randn(K,D,device='cuda')*.4).exp()+.1
            result=primitive([z,c,s],lambda z,c,s:(((z[:,:,None]-c[None,None])/s[None,None])**2).sum(-1),direct_distance)
            report['distance_checks'].append({'shape':[B,N,K,D],**result});save()
            print('DISTANCE_PASS',B,N,K,D,result['output'],flush=True)
        z=torch.ones(1,1,64,device='cuda')*100
        c=z[0].clone()+1e-3;s=torch.ones_like(c)
        ref=(((z[:,:,None]-c[None,None])/s[None,None])**2).sum(-1)
        mat=quadratic_distance(z,c,s)
        report['quadratic_cancellation_probe']={'direct':ref.item(),'matmul':mat.item(),
            'reason_not_default':'expanded squared distance can cancel for nearby large vectors'}
        report['v2_scan_checks']=[]
        for L,tiny in [(7,False),(64,False),(256,False),(7,True)]:
            u=torch.randn(2,32,L,device='cuda')
            delta=torch.rand_like(u)*(.000001 if tiny else .2)
            if tiny:delta[:,:,0]=0
            vals=[u,delta,-torch.rand(32,16,device='cuda')*4-.1,
                  torch.randn(2,16,L,device='cuda'),torch.randn(2,16,L,device='cuda')]
            r=primitive(vals,scan_without_delta_injection,scan_v2_triton,rtol=3e-3,atol=5e-4)
            report['v2_scan_checks'].append({'L':L,'tiny_zero_delta':tiny,**r});save()
            print('V2_SCAN_PASS',L,tiny,flush=True)
        data=Path(PATHS['active_split']).parent
        train,val=PlannedDataset(data,'train'),PlannedDataset(data,'val')
        tx=torch.stack([train[i][0] for i in range(8)]).cuda()
        ty=torch.stack([train[i][1] for i in range(8)]).cuda()
        vx=torch.stack([val[i][0] for i in range(2)]).cuda()
        vy=torch.stack([val[i][1] for i in range(2)]).cuda()
        criterion=baseline_api().BceDiceLoss().cuda()
        report['models']={}
        jobs=[('shared',TinyViMSharedADGBC,FastTinyViMSharedADGBC,'TinyViM_基础模型_Shared_ADGBC_最浅Skip_HAM_20261102'),
              ('v2_skip_lora',TinyViMV2SkipLoRA,FastTinyViMV2SkipLoRA,'TinyViM_V2_最浅Skip_LoRA_r8_20261101')]
        for key,oldcls,newcls,run in jobs:
            reference=out/('batch8_full_reference_'+key+'.pth')
            if not reference.exists():
                payload=torch.load(ROOT/'log'/run/'run/checkpoints/latest.pth',map_location='cpu',weights_only=False)
                torch.save({k:payload[k] for k in ('model','optimizer','epoch')},reference)
                del payload
            payload=torch.load(reference,map_location='cpu',weights_only=False)
            state=payload['model'];epoch=payload['epoch'];optimizer_state=payload['optimizer'];del payload
            # Formal runs start from the official ImageNet encoder; use that
            # same initialization for the fresh-optimizer acceptance case.
            old,new=oldcls(pretrained=True),newcls(pretrained=False)
            initial=copy.deepcopy(old.state_dict())
            fresh=network_check(old,new,initial,tx,ty,criterion,train_mode=True)
            del initial
            print('FRESH_BATCH8_PASS',key,fresh['all_gradient_relative_l2'],flush=True)
            check=network_check(old,new,state,vx,vy,criterion,optimizer_state=optimizer_state)
            trained=network_check(old,new,state,tx,ty,criterion,train_mode=True,optimizer_state=optimizer_state)
            report['models'][key]={'checkpoint_epoch':epoch,'equivalence':check,
                'fresh_training_batch8_equivalence':fresh,'training_batch8_equivalence':trained};save()
            print('NETWORK_PASS',key,check['output'],check['adam_update_relative_l2'],flush=True)
            del state,optimizer_state
            timing=paired_times(old,new,tx,ty,criterion)
            report['models'][key]['timing']=timing;save()
            print('TIMING',key,json.dumps(timing),flush=True)
            del old,new;gc.collect();torch.cuda.empty_cache()
        report['status']='passed'
    except Exception as exc:
        report['status']='failed';report['error']=repr(exc)
        raise
    finally:
        report['protected_source_hashes_unchanged']=all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in protected.items())
        files=['Models/SharedADGBC/fast_distance.py','Models/SharedADGBC/fast_model.py',
               'Models/SkipLoRA/fast_model.py','Models/SkipLoRA/triton_scan.py',
               'source/analysis/validate_acceleration.py']
        report['fast_source_sha256']={k:hashlib.sha256((ROOT/k).read_bytes()).hexdigest() for k in files}
        save()


if __name__=='__main__':
    main()

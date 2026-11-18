import os
os.environ.update(CUDA_VISIBLE_DEVICES='GPU-92ee9a11-3dfa-c3e5-0365-22c6a74bd21b',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
import sys,time,json,copy,collections
from pathlib import Path
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation');sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
import torch
from source.e1232_fourarm_screen_20261106 import entry as m
from source.e1232_matrix_20261106.candidate import build_candidate
from ham_fast_pipeline import PlannedDataset
OUT=ROOT/'operations/engineering_speed_lab/gatematrix_backward_20261106'

def main():
 torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.16);api=m.runner.baseline_api();api.seed_all(42)
 cfg=json.loads(Path('/media/8T/PuMengYu/experiments/skin_segmentation/E1232_GateMatrix_FRESH300_20261106/protocol.json').read_text())['config'];ds=PlannedDataset(Path(cfg['data']['split_file']).parent,'train');items=[ds[i] for i in range(8)];x=torch.stack([z[0] for z in items]).cuda();y=torch.stack([z[1] for z in items]).cuda()
 model=build_candidate('BT',False).cuda();initial={n:t.detach().cpu().clone() for n,t in model.state_dict().items()};crit=api.BceDiceLoss().cuda();report={'scope':'isolated BT random init true input batch8 FP32 parameters; not full epoch accuracy','runs':[],'conv_inventory':{}}
 for n,mod in model.named_modules():
  if isinstance(mod,(torch.nn.Conv2d,torch.nn.Conv1d,torch.nn.ConvTranspose2d)):
   group='encoder' if n.startswith('encoder.') else 'bottleneck' if n.startswith('decoder.e2.') else 'skip' if n.startswith('decoder.e3.') else 'decoder_other' if n.startswith('decoder.') else 'align_other'
   d=report['conv_inventory'].setdefault(group,{'layers':0,'parameters':0});d['layers']+=1;d['parameters']+=sum(p.numel() for p in mod.parameters(recurse=False))
 reference=None
 for label,dtype in [('fp32',None),('bf16',torch.bfloat16),('fp16',torch.float16),('fp32_repeat',None)]:
  model.load_state_dict(initial);model.train();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);scaler=torch.cuda.amp.GradScaler(enabled=dtype==torch.float16);row={'mode':label};times=[]
  try:
   for i in range(5):
    opt.zero_grad(set_to_none=True);torch.cuda.synchronize();t=time.perf_counter()
    with torch.autocast('cuda',dtype=dtype or torch.float16,enabled=dtype is not None):pred=model(x)
    with torch.autocast('cuda',enabled=False):loss=crit(pred.float(),y.float())
    assert bool(torch.isfinite(loss)),'nonfinite loss'
    scaler.scale(loss).backward();scaler.unscale_(opt)
    finite=bool(torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all());assert finite,'nonfinite gradient'
    if i==0:
     before=next(model.parameters()).detach().clone()
     gradients={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None};pr=pred.detach().float().cpu()
     if reference is None:reference=(pr,gradients)
     row['prediction_max_abs_diff']=float((pr-reference[0]).abs().max());row['max_parameter_gradient_abs_diff']=max(float((g-reference[1][n]).abs().max()) for n,g in gradients.items());del gradients,pr
    scaler.step(opt);scaler.update();torch.cuda.synchronize();seconds=time.perf_counter()-t
    times.append(seconds);row['last_loss']=float(loss.detach());row['scale']=scaler.get_scale()
   row.update(passed=True,steady_seconds=times[2:],warmup_seconds=times[:2],peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
  except Exception as e:row.update(passed=False,error=repr(e),completed_steps=len(times))
  report['runs'].append(row);(OUT/'amp_trial.json').write_text(json.dumps(report,indent=2));print('AMP_RESULT',json.dumps(row),flush=True);del opt,scaler;model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
 print('CONV_INVENTORY',report['conv_inventory'],flush=True)
if __name__=='__main__':main()

"""Read-only diagnostic training steps; never writes formal weights or changes LR."""
import os,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
os.environ.update(CUDA_HOME='/home/PuMengYu/anaconda3/envs/medseg',CC='/usr/bin/gcc-9',CXX='/usr/bin/g++-9',MAX_JOBS='2',TORCH_CUDA_ARCH_LIST='8.9',LIBRARY_PATH='/home/PuMengYu/.cache/skin_encoder2026/cuda121/nvidia/cuda_runtime/lib')
os.environ['CPATH']=':'.join(['/home/PuMengYu/.cache/skin_encoder2026/cuda121/nvidia/cuda_runtime/include','/home/PuMengYu/.cache/skin_encoder2026/cuda124_fixed/nvidia/cuda_cccl/include','/home/PuMengYu/anaconda3/envs/medseg/targets/x86_64-linux/include']+[f'/home/PuMengYu/anaconda3/envs/medseg/lib/python3.10/site-packages/nvidia/{s}/include' for s in ['cusparse','cusolver','cublas']])
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba'),str(Path(__file__).parent)]
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t')
import torch
import ham_fast_runner as runner
from ham_fast_pipeline import make_fast_loaders
from experiment_v1 import verify_v1_runtime
from candidates_experiment import build_model

def main():
 runner.verify_runtime=verify_v1_runtime
 config=runner.validate_protocol(runner.PROTOCOL,'tinyvim_s',check_cache=True)
 api=runner.baseline_api();api.seed_all(42);torch.set_num_threads(4)
 model=build_model('visionhope_tiny',pretrained=True).cuda().train()
 loaders,_=make_fast_loaders(config,api,workers=2,validation_batch=8)
 opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8)
 criterion=api.BceDiceLoss().cuda();records=[]
 try:
  for i,(x,y,ids,_) in enumerate(loaders['train']):
   if i>=5:break
   opt.zero_grad(set_to_none=True);pred=model(x.cuda());finite=bool(torch.isfinite(pred).all())
   record=dict(step=i+1,ids=list(ids),prediction_finite=finite);records.append(record)
   if not finite:print(json.dumps(record),flush=True);break
   loss=criterion(pred,y.cuda());loss.backward()
   bad=[n for n,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
   record.update(loss=float(loss.detach()),bad_gradients=bad)
   print(json.dumps(record),flush=True)
   if bad:break
   opt.step()
  out=ROOT/'log/八模型对比/2026候选骨干_HAM_20261104/visionhope_tiny/numerical_diagnostic.json'
  out.write_text(json.dumps(records,indent=2)+'\n')
 finally:
  for l in loaders.values():
   if hasattr(l,'close'):l.close()
if __name__=='__main__':main()

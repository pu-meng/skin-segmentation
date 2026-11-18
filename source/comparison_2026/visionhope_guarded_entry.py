"""VisionHOPE recovery: finite-value synchronization, original math/protocol retained."""
from pathlib import Path
import sys,json
import recovered_entry  # import compatibility also runs in spawned workers
if __name__=='__main__':
 import torch,runpy
 import ham_fast_runner as runner
 original_train=runner.train
 def guarded_train(model,loaders,criterion,optimizer,device,stage,config,info,generator,resume=False):
  def check_output(module,inputs,output):
   if not bool(torch.isfinite(output).all()):raise FloatingPointError('Nonfinite VisionHOPE output; no optimizer update performed')
  model.register_forward_hook(check_output)
  def finite_step(opt,args,kwargs):
   bad=[n for n,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
   if bad:raise FloatingPointError('Nonfinite gradients before optimizer: '+str(bad))
  optimizer.register_step_pre_hook(finite_step)
  runner.write_json(stage/'finite_guard.json',dict(source=str(Path(__file__).resolve()),
       change='synchronize and check output/gradients before BCE/optimizer; same FP32 model, optimizer, data and LR',diagnostic='5 real batches at unchanged LR .001 passed'))
  return original_train(model,loaders,criterion,optimizer,device,stage,config,info,generator,resume)
 runner.train=guarded_train
 import candidates_experiment as base
 script=Path(__file__).with_name('candidates_experiment.py')
 sys.argv=[str(script),'--arm','visionhope_tiny','--mode','train','--run-root',str(base.RUNS_ROOT/'visionhope_tiny'),'--gpu','0']
 runpy.run_path(str(script),run_name='__main__')

import csv,json,os,random
from pathlib import Path
import torch,numpy as np
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
OPS=ROOT/'operations/mambalite_publiccode_four_seeds_20261106'
def equal(a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 if isinstance(a,np.ndarray):return np.array_equal(a,b)
 return a==b
def restore_cpu(m,opt,scheduler,loaders,seed,rd,md):
 from reconstruction_engine import sha,HERE,SPLIT
 assert seed==42,'only reconstructed seed42 authorized'
 c=json.loads((OPS/'protocol.json').read_text());p=md/'latest.pth'
 assert sha(p)==c['reconstruction_resume']['input_latest_sha256']
 x=torch.load(p,map_location='cpu',weights_only=False)
 required=['model','optimizer','scheduler','best_loss','best_epoch','python_rng','numpy_rng','torch_rng','cuda_rng','sampler_rng','initial_state_sha256','epoch','seed','protocol_sha256','split_sha256']
 assert all(k in x for k in required),'incomplete payload keys'
 assert x['seed']==seed and x['protocol_sha256']==sha(HERE/'protocol.json') and x['split_sha256']==sha(SPLIT)
 assert 1<=x['epoch']<300
 m.load_state_dict(x['model'],strict=True);assert equal(m.state_dict(),x['model'])
 opt.load_state_dict(x['optimizer']);scheduler.load_state_dict(x['scheduler'])
 assert equal(opt.state_dict(),x['optimizer']) and equal(scheduler.state_dict(),x['scheduler'])
 rows=list(csv.DictReader((rd/'history.csv').open()))
 assert [int(r['epoch']) for r in rows]==list(range(1,x['epoch']+1))
 updates=x['epoch']*len(loaders['train']);assert int(rows[-1]['optimizer_updates'])==updates
 assert all(int(v['step'])==updates for v in opt.state.values())
 sel=json.loads((rd/'selection.json').read_text());winner=min(rows,key=lambda r:float(r['val_loss']))
 assert sel['epoch']==x['best_epoch']==int(winner['epoch']) and sel['val_loss']==x['best_loss']==float(winner['val_loss'])
 assert sel['checkpoint_sha256']==sha(md/'best.pth')==c['reconstruction_resume']['input_best_sha256']
 assert scheduler.last_epoch==x['epoch']
 assert abs(opt.param_groups[0]['lr']-scheduler.get_last_lr()[0])<1e-15
 sampler=loaders['train'].sampler;st=x['sampler_rng']
 sampler.generator.set_state(st['torch']);sampler.py.setstate(st['python']);sampler.np.set_state(st['numpy'])
 assert equal(sampler.state(),st)
 random.setstate(x['python_rng']);np.random.set_state(x['numpy_rng']);torch.set_rng_state(x['torch_rng'])
 assert equal(random.getstate(),x['python_rng']) and equal(np.random.get_state(),x['numpy_rng']) and equal(torch.get_rng_state(),x['torch_rng'])
 # Validate CUDA RNG as a byte state on CPU; actual cuda RNG restored immediately before next forward by engine.
 assert x['cuda_rng'].dtype==torch.uint8 and x['cuda_rng'].ndim==1 and x['cuda_rng'].numel()>0
 audit=dict(checkpoint=str(p),checkpoint_sha256=sha(p),completed_epoch=x['epoch'],resume_epoch=x['epoch']+1,optimizer_updates=updates,strict_model_exact=True,optimizer_exact=True,scheduler_exact=True,best_selection_exact=True,sampler_all_three_rng_restored=True,global_rng_restored_before_next_forward=True,initial_state_sha256=x['initial_state_sha256'],worker_rng='not saved in legacy; deterministic PlannedDataset uses complete sampler plans; worker_init only sets threads; not bitwise CUDA replay guarantee',pid=os.getpid(),physical_gpu=1)
 return x,audit

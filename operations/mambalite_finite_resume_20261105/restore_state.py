import csv,json,hashlib,os
from pathlib import Path
import torch
O=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def restore_cpu(m,opt,scheduler,loaders,seed,rd,md):
 assert (O/'boundary_stop.json').exists() and not Path('/proc/2534485').exists()
 p=O/'boundary_latest.pth';x=torch.load(p,map_location='cpu',weights_only=False)
 from candidate_model import HERE
 from experiment import SPLIT
 assert x['seed']==seed and x['protocol_sha256']==sha(HERE/'protocol.json') and x['split_sha256']==sha(SPLIT)
 assert x['epoch']==48
 m.load_state_dict(x['model'],strict=True)
 assert all(torch.equal(v,x['model'][k]) for k,v in m.state_dict().items())
 opt.load_state_dict(x['optimizer']);scheduler.load_state_dict(x['scheduler'])
 def equal(a,b):
  if torch.is_tensor(a):return torch.equal(a,b)
  if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
  if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(v,w) for v,w in zip(a,b))
  return a==b
 assert equal(opt.state_dict(),x['optimizer']) and equal(scheduler.state_dict(),x['scheduler'])
 rows=list(csv.DictReader((rd/'history.csv').open()))
 assert [int(r['epoch']) for r in rows]==list(range(1,x['epoch']+1))
 updates=int(rows[-1]['optimizer_updates']);assert updates==x['epoch']*len(loaders['train'])
 assert all(int(v['step'])==updates for v in opt.state.values())
 sel=json.loads((rd/'selection.json').read_text());assert sel['epoch']==x['best_epoch'] and sel['val_loss']==x['best_loss'] and sel['checkpoint_sha256']==sha(md/'best.pth')
 sampler=loaders['train'].sampler;st=x['sampler_rng'];sampler.generator.set_state(st['torch']);sampler.py.setstate(st['python']);sampler.np.set_state(st['numpy'])
 # Actual dataset has no RNG calls; all augmentation is encoded by sampler in parent.
 # Worker RNG / loader seed generator were not saved by original engine, but never affect returned samples.
 audit=dict(checkpoint=str(p),checkpoint_sha256=sha(p),completed_epoch=x['epoch'],resume_epoch=x['epoch']+1,optimizer_updates=updates,optimizer_state_tensors=len(opt.state),strict_model_exact=True,optimizer_exact=True,scheduler_exact=True,best_selection_exact=True,sampler_all_three_rng_restored=True,global_rng_restored_before_next_forward=True,worker_rng='original payload did not save worker RNG or loader generator; dataset deterministic from saved sampler plans, worker_init only sets threads; no worker RNG-dependent samples',initial_state_sha256=x['initial_state_sha256'],pid=os.getpid(),physical_gpu=1,engineering_manifest_sha256=sha(O/'revision_manifest.json'))
 return x,audit

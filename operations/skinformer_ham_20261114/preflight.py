"""SkinFormer local adaptation: isolated real-HAM batch16 GPU preflight."""
import os,sys,json,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation'); OPS=ROOT/'operations/skinformer_ham_20261114';VENDOR=ROOT/'operations/skinformer_adapt_20261114/vendor'
sys.path[:0]=[str(VENDOR),str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
import torch
from torch.utils.data import DataLoader
from Models.networks.net import skinformer
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t')
import baseline_runner_seed42 as api
HAM=ROOT/'data/processed/ham10000_group42_dedup_20261023'

torch.set_num_threads(4);torch.manual_seed(42);torch.cuda.set_device(0)
ds=api.HAMDataset(HAM,'val'); x,y,*_=next(iter(DataLoader(ds,batch_size=16,shuffle=False,num_workers=0)))
model=skinformer(SimpleNamespace(out_size=256),n_classes=2).cuda().train();opt=torch.optim.Adam(model.parameters(),lr=.0002,weight_decay=1e-8)
x=x.cuda();y=y.cuda().reshape(16,1,256,256);torch.cuda.reset_peak_memory_stats();t=time.monotonic();out=model(x);p=out[:,1:2]
loss=1-(2*(p*y).sum()+1e-6)/((p.square()+y.square()).sum()+1e-6);assert bool(torch.isfinite(loss));loss.backward()
grads=[q.grad for q in model.parameters() if q.grad is not None];assert grads and all(torch.isfinite(g).all() for g in grads)
opt.step();torch.cuda.synchronize();assert torch.isfinite(out).all()
result={'passed':True,'batch_size':16,'dataset':'HAM10000 validation, no test','input_shape':list(x.shape),'output_shape':list(out.shape),'loss':float(loss.detach()),'seconds':time.monotonic()-t,'peak_allocated_mib':round(torch.cuda.max_memory_allocated()/2**20,1),'peak_reserved_mib':round(torch.cuda.max_memory_reserved()/2**20,1),'pid':os.getpid(),'device':torch.cuda.get_device_name(0),'gradients_finite':True,'warning':'locally reconstructed encoder and upsampling layers; author equivalence unverified'}
(OPS/'gpu_preflight.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False),flush=True)

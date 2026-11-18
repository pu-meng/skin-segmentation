import os,sys,json
from pathlib import Path
O=Path(__file__).resolve().parent
sys.path.insert(0,str(O));sys.path.insert(0,'/home/PuMengYu/experiments/skin_segmentation/source/mambalite_paper_candidate_20261105')
import experiment as e
import torch
from restore_state import restore_cpu
e.verify_sources();m=e.build_model();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=50,eta_min=1e-5)
x,a=restore_cpu(m,opt,sch,e.make_loaders(42),42,e.run_path(42),e.model_path(42));assert not torch.cuda.is_initialized()
a['CUDA_initialized']=False;e.save(O/'cpu_restore_acceptance.json',a);print(json.dumps(a),flush=True)

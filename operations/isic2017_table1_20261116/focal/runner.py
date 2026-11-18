import importlib.util,sys,json
from pathlib import Path
import torch
R=Path('/home/PuMengYu/experiments/skin_segmentation'); O=Path('/home/PuMengYu/experiments/skin_segmentation/operations/isic2017_table1_20261116/focal'); OUT=Path('/home/PuMengYu/experiments/skin_segmentation/log/ISIC2017_表1_去重_20261116/FocalTransNet'); DATA=Path('/home/PuMengYu/experiments/skin_segmentation/data/processed/ISIC2017_去重_seed42_20261116')
s=importlib.util.spec_from_file_location('focal_official',R/'operations/focaltransnet_isic_20261114/run.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.CACHE=Path('/home/PuMengYu/experiments/skin_segmentation/operations/isic2017_table1_20261116/cache');m.SPLIT=DATA/'split.csv';m.CONFIG_PATH=O/'config.json'
phase=sys.argv[1]
if phase=='check':
 for name,n in [('train',1997),('val',149),('test',600)]:assert len(m.IsicSplit(name,False))==n
 m.IsicSplit('train',True)[0]
 print('DATA_AND_AUGMENTATION_OK',flush=True)
elif phase=='preflight':
 m.seed_everything(1234);torch.set_num_threads(4); model=m.build_model().cuda().train();loader=m.make_loader('train',True,json.loads(m.CONFIG_PATH.read_text()));x,y,_=next(iter(loader));assert len(x)==6;opt=torch.optim.AdamW(model.parameters(),lr=6e-5,weight_decay=.03);opt.zero_grad();out=model(x.cuda());loss=m.OfficialIsicLoss()(out,y.cuda())[0];assert torch.isfinite(loss);loss.backward();g=[p.grad for p in model.parameters() if p.grad is not None];assert g and all(torch.isfinite(v).all() for v in g);opt.step();print('PREFLIGHT_PASS',float(loss),torch.cuda.max_memory_reserved()/2**20,flush=True)
elif phase=='train':m.train(OUT,torch.device('cuda:0'))
elif phase=='test':
 import subprocess
 m.test(OUT,torch.device('cuda:0'))
 torch.cuda.empty_cache()
 subprocess.run([sys.executable,'-B','-u',str(O/'audit.py')],check=True)
else:raise ValueError(phase)

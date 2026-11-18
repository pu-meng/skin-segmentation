import importlib.util,sys,json
from pathlib import Path
import torch
R=Path('/home/PuMengYu/experiments/skin_segmentation'); O=R/'operations/ph2_focal_author_20261115'; OUT=Path('/media/8T/PuMengYu/experiments/skin_segmentation/PH2_FOCALTRANSNET_AUTHOR_seed1234_20261115'); DATA=R/'data/processed/PH2_256_ratio_7_1_2_seed42_20261115'
s=importlib.util.spec_from_file_location('focal_official',R/'operations/focaltransnet_isic_20261114/run.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.CACHE=O/'cache';m.SPLIT=DATA/'split.csv';m.CONFIG_PATH=O/'config.json'
phase=sys.argv[1]
if phase=='check':
 for name,n in [('train',140),('val',20),('test',40)]:assert len(m.IsicSplit(name,False))==n
 m.IsicSplit('train',True)[0]
 print('DATA_AND_AUGMENTATION_OK',flush=True)
elif phase=='preflight':
 m.seed_everything(1234);torch.set_num_threads(4); model=m.build_model().cuda().train();loader=m.make_loader('train',True,json.loads(m.CONFIG_PATH.read_text()));x,y,_=next(iter(loader));assert len(x)==6;opt=torch.optim.AdamW(model.parameters(),lr=6e-5,weight_decay=.03);opt.zero_grad();out=model(x.cuda());loss=m.OfficialIsicLoss()(out,y.cuda())[0];assert torch.isfinite(loss);loss.backward();g=[p.grad for p in model.parameters() if p.grad is not None];assert g and all(torch.isfinite(v).all() for v in g);opt.step();print('PREFLIGHT_PASS',float(loss),torch.cuda.max_memory_reserved()/2**20,flush=True)
elif phase=='train':m.train(OUT,torch.device('cuda:0'))
elif phase=='test':m.test(OUT,torch.device('cuda:0'))
else:raise ValueError(phase)

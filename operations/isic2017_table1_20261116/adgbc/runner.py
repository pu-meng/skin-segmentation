import importlib.util,sys,json
from pathlib import Path
R=Path('/home/PuMengYu/experiments/skin_segmentation');S=R/'source/adgbc_repro_20261110';D=Path('/home/PuMengYu/experiments/skin_segmentation/operations/isic2017_table1_20261116/cache');O=Path('/home/PuMengYu/experiments/skin_segmentation/log/ISIC2017_表1_去重_20261116/AD-GBC');sys.path.insert(0,str(S))
phase=sys.argv[1]
def module(n):
 spec=importlib.util.spec_from_file_location('ph2_'+n,S/(n+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.DATASETS['ISIC2017']={'images_npy':str(D/'images.npy'),'masks_npy':str(D/'masks.npy'),'split_csv':str(D/'split.csv'),'counts':{'train':1997,'val':149,'test':600},'size':256};return m
if phase=='check':
 m=module('train_adgbc')
 for s,n in [('train',1997),('val',149),('test',600)]:assert len(m.build_loaders(m.DATASETS['ISIC2017'],s,8,0,s=='train').dataset)==n
 print('DATA_OK',flush=True)
elif phase=='preflight':
 import torch
 m=module('train_adgbc');torch.set_num_threads(4);m.seed_torch(1029);model=m.archs_GBC.GBC_Rolling_Unet_L(num_classes=1,input_channels=3,deep_supervision=False,gbc_num_balls=32,gbc_proj_dim=None,use_diag_cov=True,tau=1.).cuda().train();lossfn=m.losses.BCEDiceWithGeometryLoss(div_weight=.01,scale_weight=.01).cuda();other=[];gbc=[]
 for n,p in model.named_parameters():
  if p.requires_grad:(gbc if 'centers' in n.lower() or 'gbc' in n.lower() else other).append(p)
 opt=torch.optim.Adam([{'params':other,'lr':.0001,'weight_decay':.0001},{'params':gbc,'lr':.01,'weight_decay':.0001}]);x,y,_=next(iter(m.build_loaders(m.DATASETS['ISIC2017'],'train',8,0,True)));assert len(x)==8;out=model(x.cuda());loss=lossfn(out,y.cuda(),model);assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None);opt.step();print('PREFLIGHT_PASS',float(loss.detach()),torch.cuda.max_memory_reserved()/2**20,flush=True)
elif phase=='train':
 assert not O.exists();m=module('train_adgbc');sys.argv=['train_adgbc','--dataset','ISIC2017','--out',str(O),'--gpu','0'];m.main();p=O/'protocol.json';v=json.loads(p.read_text());v['deviations']=['ISIC2017 deduplicated 1997/149/600 split and 256px cache used','ImageNet normalization once: project adapter differs from double-normalized vendor dataset'];v.pop('paper_report',None);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
elif phase in ('test','tta'):
 m=module('evaluate_final' if phase=='test' else 'evaluate_tta');sys.argv=['evaluate','--dataset','ISIC2017','--out',str(O),'--gpu','0'];m.main()
elif phase in ('audit','audit_tta'):
 m=module('audit_adgbc');folder='ours' if phase=='audit' else 'ours_tta';sys.argv=['audit','--dataset','ISIC2017','--product-dir',str(O/'final_test'/folder),'--ckpt',str(O/'best.pth')];m.main();assert json.loads((O/'final_test'/folder/'artifact_audit.json').read_text())['complete'],'artifact audit failed'
else:raise ValueError(phase)

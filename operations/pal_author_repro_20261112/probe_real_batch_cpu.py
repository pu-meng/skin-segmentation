#!/usr/bin/env python3
from __future__ import annotations
import contextlib, os, sys, types
from pathlib import Path
from types import SimpleNamespace
BASE=Path('/home/PuMengYu/experiments/skin_segmentation')
OP=BASE/'operations/pal_author_repro_20261112'
VENDOR=OP/'vendor/PAL'
DATA=Path('/media/8T/PuMengYu/experiments/skin_segmentation/PAL_AUTHOR_ISIC2018_DATA_20261112')
sys.path[:0]=[str(OP/'vendor_deps'),str(VENDOR),str(VENDOR/'src')]
os.chdir(VENDOR/'src')
w=types.ModuleType('wandb');w.init=lambda **k:None;w.log=lambda *a,**k:None;w.finish=lambda *a,**k:None;sys.modules['wandb']=w
a=types.ModuleType('apex');a.amp=types.SimpleNamespace(initialize=lambda m,o,**k:(m,o),scale_loss=lambda l,o:contextlib.nullcontext(l));sys.modules['apex']=a
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
orig_tcuda=torch.Tensor.cuda;orig_mcuda=torch.nn.Module.cuda
torch.Tensor.cuda=lambda self,*a,**k:self
torch.nn.Module.cuda=lambda self,*a,**k:self
try:
    import our_train
    from our_model import Model
    from utils.dataset import ISIC2018Data
    from utils.losses import structure_loss,intra_att_loss
    args=SimpleNamespace(backbone='pvt',patchsize=352,snapshot=None,mask_dim=256,num_queries=2,num_classes=2,cls_attributes=10,dec_layers=9,shareW=True,fuse_mode='conv',topk=3,sample_num=15,tau=1.0,decorrtype='queryfeat',intra_div=1.0,clsw=1.0,mcsw=2.0,fusew=2.0,loadbal=1.0)
    torch.manual_seed(1337)
    model=Model(args).train()
    loader=DataLoader(ISIC2018Data(str(DATA),'train',352),batch_size=1,shuffle=False,num_workers=0)
    image,mask=next(iter(loader));image=image.float();mask=mask.float()
    size=440
    image=F.interpolate(image,size=(size,size),mode='bilinear',align_corners=True)
    mask=F.interpolate(mask.unsqueeze(1),size=(size,size),mode='bilinear',align_corners=True).squeeze(1)
    out=model(image,mask)
    feat=torch.cat([m.weight.unsqueeze(0) for m in model.mask2former.query_feat],dim=0)
    gram=torch.einsum('ckd,cqd->ckq',feat,feat)
    ld=intra_att_loss(gram); lb=our_train.cv_squared(out['topk_mask'].sum(0))
    cls=F.interpolate(out['cls_prob'],size=mask.shape[1:],mode='bilinear',align_corners=True)
    lc=structure_loss(cls,mask,att='interior')
    lm=structure_loss(out['mcs_prob'],mask,att='boundary')
    lf=structure_loss(out['fuse_prob'],mask,att='boundary')
    loss=lc+2*lm+2*lf+ld+lb
    assert torch.isfinite(loss)
    loss.backward()
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    print('PAL_REAL_TRAIN_FORWARD_BACKWARD_OK', 'batch',image.shape[0],'input',tuple(image.shape),'loss',float(loss),'grad_tensors',len(grads),'params',sum(p.numel() for p in model.parameters()))
finally:
    torch.Tensor.cuda=orig_tcuda;torch.nn.Module.cuda=orig_mcuda

import os, sys, json
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
R=Path('/home/PuMengYu/experiments/skin_segmentation')
sys.path.insert(0,str(R/'references/code/FuseUNet/FuseUNet'))
sys.path.insert(0,str(R/'operations/fuseunet_author_repro_20261112/deps'))
import torch
from torch import nn
from nnunetv2.nets.unet import PlainConvUNet
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from types import SimpleNamespace
torch.set_num_threads(1)
torch.manual_seed(42)
t=nnUNetTrainer.__new__(nnUNetTrainer)
t.device=torch.device('cpu');t.grad_scaler=None;t.enable_deep_supervision=False
t.label_manager=SimpleNamespace(has_regions=False,has_ignore_label=False)
t.network=PlainConvUNet(input_channels=3,n_stages=6,features_per_stage=(32,64,128,256,320,320),conv_op=nn.Conv2d,kernel_sizes=(3,)*6,strides=(1,2,2,2,2,2),n_conv_per_stage=(2,)*6,num_classes=2,n_conv_per_stage_decoder=(2,)*5,conv_bias=True,norm_op=nn.InstanceNorm2d,norm_op_kwargs={'eps':1e-5,'affine':True},dropout_op=None,dropout_op_kwargs=None,nonlin=nn.LeakyReLU,nonlin_kwargs={'inplace':True},deep_supervision=False)
t.loss=DC_and_CE_loss({'batch_dice':False,'smooth':1e-5,'do_bg':False,'ddp':False},{},weight_ce=1,weight_dice=1,dice_class=MemoryEfficientSoftDiceLoss)
t.initial_lr=0.001;t.weight_decay=3e-5;t.num_epochs=1000
t.optimizer,t.lr_scheduler=t.configure_optimizers()
batch={'data':torch.rand(1,3,64,64),'target':torch.randint(0,2,(1,1,64,64)).float()}
p=next(t.network.parameters()); before=p.detach().clone(); out=t.train_step(batch)
with torch.no_grad(): val=t.validation_step(batch)
result={'kind':'synthetic CPU direct vendor train_step and validation_step, constructed trainer without init; not full trainer/dataloader','train_loss':float(out['loss']),'val_loss':float(val['loss']),'changed':bool(torch.any(p.detach()!=before)),'finite':bool(torch.isfinite(p).all()),'tp':val['tp_hard'].tolist(),'fp':val['fp_hard'].tolist(),'fn':val['fn_hard'].tolist(),'cuda_initialized':torch.cuda.is_initialized(),'optimizer':str(t.optimizer),'model_parameters':sum(p.numel() for p in t.network.parameters())}
assert result['changed'] and result['finite'] and not result['cuda_initialized']
p=R/'operations/fuseunet_author_repro_20261112/cpu_trainer_probe.json';p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

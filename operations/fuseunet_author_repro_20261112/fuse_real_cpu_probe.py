import os,sys,json
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
R=Path('/home/PuMengYu/experiments/skin_segmentation'); O=Path(__file__).resolve().parent
sys.path[:0]=[str(O/'deps'),str(R/'references/code/FuseUNet/FuseUNet'),str(O)]
import torch,numpy as np
from torch import nn
from types import SimpleNamespace
from fuse_skin_adapter import SkinCases,CONFIG
from nnunetv2.nets.unet import PlainConvUNet
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

torch.set_num_threads(1); torch.manual_seed(42); np.random.seed(42)
label=SimpleNamespace(all_labels=[0,1],has_ignore_label=False,has_regions=False)
results={}
for name in CONFIG:
 t=nnUNetTrainer.__new__(nnUNetTrainer); t.device=torch.device('cpu'); t.grad_scaler=None; t.enable_deep_supervision=False; t.label_manager=label
 t.network=PlainConvUNet(input_channels=3,n_stages=6,features_per_stage=(32,64,128,256,320,320),conv_op=nn.Conv2d,kernel_sizes=(3,)*6,strides=(1,2,2,2,2,2),n_conv_per_stage=(2,)*6,num_classes=2,n_conv_per_stage_decoder=(2,)*5,conv_bias=True,norm_op=nn.InstanceNorm2d,norm_op_kwargs={'eps':1e-5,'affine':True},dropout_op=None,dropout_op_kwargs=None,nonlin=nn.LeakyReLU,nonlin_kwargs={'inplace':True},deep_supervision=False)
 t.loss=DC_and_CE_loss({'batch_dice':False,'smooth':1e-5,'do_bg':False,'ddp':False},{},weight_ce=1,weight_dice=1,dice_class=MemoryEfficientSoftDiceLoss)
 t.initial_lr=.001; t.weight_decay=3e-5; t.num_epochs=1000; t.optimizer,t.lr_scheduler=t.configure_optimizers()
 ds=SkinCases(name,'train'); vs=SkinCases(name,'val')
 train=nnUNetDataLoader2D(ds,1,(362,362),(256,256),label,oversample_foreground_percent=.33)
 val=nnUNetDataLoader2D(vs,1,(256,256),(256,256),label,oversample_foreground_percent=.33)
 tf=t.get_training_transforms((256,256),{'x':(-np.pi,np.pi),'y':(0,0),'z':(0,0)},None,(0,1),False,use_mask_for_norm=[False]*3,foreground_labels=[1])
 vf=t.get_validation_transforms(None,foreground_labels=[1])
 batch=next(SingleThreadedAugmenter(train,tf)); vb=next(SingleThreadedAugmenter(val,vf))
 p=next(t.network.parameters()); old=p.detach().clone(); out=t.train_step(batch); v=t.validation_step(vb)
 result={'train_loss':float(out['loss']),'val_loss':float(v['loss']),'changed':bool(torch.any(p.detach()!=old)), 'input_min':float(batch['data'].min()),'input_max':float(batch['data'].max()),'target_values':torch.unique(batch['target']).tolist(),'cuda_initialized':torch.cuda.is_initialized()}
 assert result['changed'] and np.isfinite(result['train_loss']) and np.isfinite(result['val_loss']) and not result['cuda_initialized']
 results[name]=result; print(name,result,flush=True)
(O/'real_cpu_acceptance.json').write_text(json.dumps(results,indent=2)+'\n')

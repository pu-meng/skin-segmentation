import os,sys,json,time,csv,hashlib,traceback
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
OP=ROOT/'operations/fuseunet_author_repro_20261112'
VENDOR=ROOT/'references/code/FuseUNet/FuseUNet'
sys.path[:0]=[str(OP/'deps'),str(VENDOR),str(OP)]
import nnunetv2.paths as paths
paths.nnUNet_raw=str(OP/'nnunet_raw'); paths.nnUNet_preprocessed=str(OP/'nnunet_preprocessed'); paths.nnUNet_results='/media/8T/PuMengYu/experiments/skin_segmentation/FUSEUNET_AUTHOR_FRESH_20261112'
from batchgenerators.utilities.file_and_folder_operations import load_json,save_json,maybe_mkdir_p
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from torch import nn

CACHE={
 'HAM10000':(ROOT/'data/processed/ham10000_group42',ROOT/'data/processed/ham10000_group42_dedup_20261023/split.csv',901),
 'ISIC2018':(ROOT/'data/processed/isic2018_256_dedup_seed42_20261031',ROOT/'data/processed/isic2018_256_dedup_seed42_20261031/split.csv',902)}

class SkinCases(nnUNetDataset):
 def __init__(self,name,split):
  assert split in ('train','val')
  root,sp,_=CACHE[name]; df=pd.read_csv(sp); df=df[df.split==split]
  self.dataset=dict(zip(df.image_id.astype(str),df.cache_index.astype(int)))
  if not self.dataset: raise ValueError(f'empty {name}/{split}')
  self.images=np.load(root/'images.npy',mmap_mode='r'); self.masks=np.load(root/'masks.npy',mmap_mode='r')
 def load_case(self,key):
  i=self.dataset[key]; x=np.asarray(self.images[i],dtype=np.float32).transpose(2,0,1)[:,None]/255.
  y=(np.asarray(self.masks[i])>127.5).astype(np.int16)[None,None]
  loc=DefaultPreprocessor._sample_foreground_locations(y,[1])[1]
  if len(loc)==0: loc=np.empty((0,4),dtype=np.int64)
  return x,y,{'class_locations':{1:loc}}

class FuseSkinTrainer(nnUNetTrainer):
 def __init__(self,plans,configuration,fold,dataset_json,unpack_dataset=True,device=torch.device('cuda')):
  super().__init__(plans,configuration,fold,dataset_json,unpack_dataset,device)
  self.dataset_name={'Dataset901_HAM10000':'HAM10000','Dataset902_ISIC2018':'ISIC2018'}[self.plans_manager.dataset_name]
  self.oversample_foreground_percent=.33; self.num_epochs=1000
  self.num_iterations_per_epoch=250; self.num_val_iterations_per_epoch=50
  self.enable_deep_supervision=False
 def get_tr_and_val_datasets(self):
  return SkinCases(self.dataset_name,'train'),SkinCases(self.dataset_name,'val')
 def on_train_start(self):
  if not self.was_initialized: self.initialize()
  maybe_mkdir_p(self.output_folder)
  self.set_deep_supervision_enabled(False); self.print_plans()
  self.inference_allowed_mirroring_axes=(0,1)
  self.dataloader_train,self.dataloader_val=self.get_dataloaders()
  save_json(self.plans_manager.plans,os.path.join(self.output_folder_base,'plans.json'))
  save_json(self.dataset_json,os.path.join(self.output_folder_base,'dataset.json'))
  self._save_debug_information()

def run(dataset_name,out):
 root,split_path,did=CACHE[dataset_name]; dataset_tag=f'Dataset{did:03d}_{dataset_name}'
 plans=load_json(str(OP/'nnunet_preprocessed'/dataset_tag/'FuseUNetPlans.json'))
 dataset_json=load_json(str(OP/'nnunet_raw'/dataset_tag/'dataset.json'))
 split=pd.read_csv(split_path); assert set(split.split)=={'train','val','test'}
 out=Path(out); out.mkdir(parents=True,exist_ok=True)
 protocol={'dataset':dataset_name,'split_csv':str(split_path),'split_sha256':hashlib.sha256(split_path.read_bytes()).hexdigest(),'plans':str(OP/'nnunet_preprocessed'/dataset_tag/'FuseUNetPlans.json'),'batch_size':plans['configurations']['2d']['batch_size'],'patch_size':plans['configurations']['2d']['patch_size'],'epochs':1000,'iterations_per_epoch':250,'validation_iterations_per_epoch':50,'optimizer':'vendor SGD lr=0.001, momentum=0.99, Nesterov, weight_decay=3e-5; vendor PolyLR','selection':'vendor EMA foreground pseudo-Dice, val only','test':'only after training, fixed checkpoint_best; direct full 256x256 inference','deviations':['cached project train/val/test split used','256x256 cached RGB images/masks; train/val use original vendor 2D sampler and transforms','synthetic unit pixel spacing was used only for planning; no physical image spacing exists in PNG cache']}
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
 trainer=FuseSkinTrainer(plans,'2d',0,dataset_json,unpack_dataset=False,device=torch.device('cuda'))
 print('TRAINER_READY',os.getpid(),dataset_name,'batch',trainer.configuration_manager.batch_size,'patch',trainer.configuration_manager.patch_size,flush=True)
 trainer.run_training()
 ckpt=torch.load(os.path.join(trainer.output_folder,'checkpoint_best.pth'),map_location='cuda',weights_only=False)
 trainer.network.load_state_dict(ckpt['network_weights']); trainer.network.eval()
 test=split[split.split=='test']; imgs=np.load(root/'images.npy',mmap_mode='r'); masks=np.load(root/'masks.npy',mmap_mode='r')
 tp=fp=fn=tn=0; per=[]
 with torch.no_grad():
  for row in test.itertuples(index=False):
   x=np.asarray(imgs[int(row.cache_index)],dtype=np.float32).transpose(2,0,1)/255.
   y=np.asarray(masks[int(row.cache_index)])>127.5
   logits=trainer.network(torch.from_numpy(x[None]).cuda()); pred=logits.argmax(1)[0].cpu().numpy()==1
   a=int((pred&y).sum()); b=int((pred&~y).sum()); c=int((~pred&y).sum()); d=int((~pred&~y).sum())
   tp+=a;fp+=b;fn+=c;tn+=d;per.append([row.image_id,a,b,c,d])
 with (out/'test_per_image.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['image_id','tp','fp','fn','tn']);w.writerows(per)
 summary={'split':'test','n_images':len(test),'checkpoint':'checkpoint_best.pth','best_epoch':int(ckpt['current_epoch']),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'global_dice':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,'global_iou':tp/(tp+fp+fn) if tp+fp+fn else 1.}
 (out/'test_summary.json').write_text(json.dumps(summary,indent=2));print('TEST_COMPLETE',json.dumps(summary),flush=True)
if __name__=='__main__':
 try: run(sys.argv[1],sys.argv[2])
 except BaseException: traceback.print_exc(); raise

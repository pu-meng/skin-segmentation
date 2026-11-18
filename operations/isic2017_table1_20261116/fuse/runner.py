import os,sys,json,csv,hashlib,importlib,importlib.util
from pathlib import Path
import numpy as np
import torch
from PIL import Image
R=Path('/home/PuMengYu/experiments/skin_segmentation');O=Path('/home/PuMengYu/experiments/skin_segmentation/operations/isic2017_table1_20261116/fuse');D=Path('/home/PuMengYu/experiments/skin_segmentation/operations/isic2017_table1_20261116/cache');ROOT=Path('/home/PuMengYu/experiments/skin_segmentation/log/ISIC2017_表1_去重_20261116/FuseUNet');phase=sys.argv[1]
spec=importlib.util.spec_from_file_location('fuse_author_adapter',R/'operations/fuseunet_author_repro_20261112/train_fuseunet.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.CACHE['ISIC2018']=(D,D/'split.csv',902)
mod=importlib.import_module('nnunetv2.training.nnUNetTrainer.nnUNetTrainer');runroot=ROOT/('preflight_nnunet' if phase=='preflight' else 'nnunet');mod.nnUNet_results=str(runroot);m.paths.nnUNet_results=str(runroot)
plans=m.load_json(str(m.OP/'nnunet_preprocessed/Dataset902_ISIC2018/FuseUNetPlans.json'));dj=m.load_json(str(m.OP/'nnunet_raw/Dataset902_ISIC2018/dataset.json'));assert plans['configurations']['2d']['batch_size']==49
if phase=='check':
 for split,n in [('train',1997),('val',149)]:assert len(m.SkinCases('ISIC2018',split).dataset)==n
 print('DATA_AND_BATCH49_OK',flush=True)
elif phase=='preflight':
 torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(4);t=m.FuseSkinTrainer(plans,'2d',0,dj,unpack_dataset=False,device=torch.device('cuda'));t.on_train_start();t.network.train();batch=next(t.dataloader_train);assert batch['data'].shape[0]==49;result=t.train_step(batch);assert np.isfinite(result['loss']);print('PREFLIGHT_PASS',result,torch.cuda.max_memory_reserved()/2**20,flush=True)
elif phase=='train':
 assert not runroot.exists();torch.manual_seed(42);torch.cuda.manual_seed_all(42);m.run('ISIC2018',ROOT/'results'); p=ROOT/'results/protocol.json'; v=json.loads(p.read_text()); v['dataset']='ISIC2017_deduplicated'; v['dataset_internal_plan_key']='ISIC2018'; p.write_text(json.dumps(v,indent=2)+'\n'); print('TRAIN_AND_NONE_DONE',flush=True)
elif phase=='test':
 matches=list(runroot.rglob('checkpoint_best.pth'));assert len(matches)==1,matches;ck=matches[0];t=m.FuseSkinTrainer(plans,'2d',0,dj,unpack_dataset=False,device=torch.device('cuda'));t.initialize();p=torch.load(ck,map_location='cpu',weights_only=False);t.network.load_state_dict(p['network_weights']);net=t.network.cuda().eval();rows=list(csv.DictReader((D/'split.csv').open()));rows=[r for r in rows if r['split']=='test'];assert len(rows)==600;imgs=np.load(D/'images.npy',mmap_mode='r');masks=np.load(D/'masks.npy',mmap_mode='r');digest=hashlib.sha256(ck.read_bytes()).hexdigest()
 for mode in ('none','mirror4'):
  dest=ROOT/'final_test'/mode;predir=dest/'predictions';predir.mkdir(parents=True,exist_ok=False);out=[];total=np.zeros(4,dtype=np.int64)
  with torch.inference_mode():
   for row in rows:
    i=int(row['cache_index']);x=torch.from_numpy(np.asarray(imgs[i],dtype=np.float32).transpose(2,0,1).copy()[None]/255).cuda();y=np.asarray(masks[i])>127.5;z=net(x)
    if mode=='mirror4':z=(z+net(x.flip(-1)).flip(-1)+net(x.flip(-2)).flip(-2)+net(x.flip((-2,-1))).flip((-2,-1)))/4
    pred=(z.argmax(1)[0].cpu().numpy()==1);cnt=np.array([(pred&y).sum(),(pred&~y).sum(),(~pred&y).sum(),(~pred&~y).sum()],dtype=np.int64);total+=cnt;Image.fromarray(pred.astype(np.uint8)*255).save(predir/(row['image_id']+'.png'));out.append([row['image_id'],*map(int,cnt)])
  with (dest/'per_image_metrics.csv').open('w',newline='') as f:w=csv.writer(f);w.writerow(['image_id','tp','fp','fn','tn']);w.writerows(out)
  tp,fp,fn,tn=map(int,total);summary={'dataset':'ISIC2017_deduplicated','model':'FuseUNet','images':600,'selected_epoch':int(p['current_epoch']),'checkpoint_sha256':digest,'tta':mode=='mirror4','global':{'dice':2*tp/max(1,2*tp+fp+fn),'iou':tp/max(1,tp+fp+fn),'tp':tp,'fp':fp,'fn':fn,'tn':tn},'test_used_for_checkpoint_selection':False};(dest/'summary.json').write_text(json.dumps(summary,indent=2));(dest/'artifact_audit.json').write_text(json.dumps({'complete':len(list(predir.glob('*.png')))==600 and len(out)==600,'prediction_count':600,'metric_count':len(out),'ids_match_frozen_test_split':True,'checkpoint_sha256':digest},indent=2));print('TEST',mode,summary['global']['dice'],flush=True)
else:raise ValueError(phase)

if phase=='test':
 for mode in ('none','mirror4'):
  a=json.loads((ROOT/'final_test'/mode/'artifact_audit.json').read_text()); assert a['complete'] and a['metric_count']==600
 (ROOT/'test_complete.json').write_text(json.dumps({'complete':True,'test_images_per_mode':600,'dataset':'ISIC2017_deduplicated'},indent=2)+'\n')

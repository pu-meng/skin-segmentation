#!/usr/bin/env python3
"""FocalTransNet ISIC recipe adapted to the frozen project HAM split."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path('/home/PuMengYu/experiments/skin_segmentation')
OPS = ROOT/'operations/focaltransnet_ham_20261114'
BASE = ROOT/'operations/focaltransnet_isic_20261114/run.py'
CACHE = ROOT/'data/processed/ham10000_group42'
SPLIT = ROOT/'data/processed/ham10000_group42_dedup_20261023/split.csv'
RUN = Path('/media/8T/PuMengYu/experiments/skin_segmentation/FocalTransNet_HAM_FRESH100_20261114')
spec = importlib.util.spec_from_file_location('focal_project_isic_adapter', BASE)
focal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(focal)

def read(path):
    return json.loads(Path(path).read_text())

def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)

def split_rows():
    with SPLIT.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    with (CACHE/'split.csv').open(newline='') as stream:
        original = {row['image_id']: row for row in csv.DictReader(stream)}
    if len({row['image_id'] for row in rows}) != len(rows):
        raise RuntimeError('duplicate ID in HAM split')
    if any(row['image_id'] not in original or row['cache_index'] != original[row['image_id']]['cache_index'] for row in rows):
        raise RuntimeError('HAM cache and frozen split identity mismatch')
    manifest = read(CACHE/'cache_complete.json')
    if focal.sha256(CACHE/'split.csv') != manifest['split_sha256']:
        raise RuntimeError('HAM cache split differs from its manifest')
    expected = {'train': 6983, 'val': 1010, 'test': 2020}
    if {key: sum(row['split'] == key for row in rows) for key in expected} != expected:
        raise RuntimeError('unexpected HAM split counts')
    return rows

class HamSplit(focal.IsicSplit):
    def __init__(self, split, training):
        if split not in ('train', 'val', 'test'):
            raise ValueError(split)
        self.rows = [row for row in split_rows() if row['split'] == split]
        self.images = np.load(CACHE/'images.npy', mmap_mode='r')
        self.masks = np.load(CACHE/'masks.npy', mmap_mode='r')
        self.training = training
        self.augmenter = None
        if training:
            if not hasattr(np, 'sctypes'):
                np.sctypes = {'int': [np.int8,np.int16,np.int32,np.int64],
                              'uint': [np.uint8,np.uint16,np.uint32,np.uint64],
                              'float': [np.float16,np.float32,np.float64],
                              'complex': [np.complex64,np.complex128], 'others': [bool,object,bytes,str,np.void]}
            import imgaug.augmenters as iaa
            self.augmenter = iaa.SomeOf((0,4), [
                iaa.Flipud(.5), iaa.Fliplr(.5), iaa.AdditiveGaussianNoise(scale=.005*255),
                iaa.GaussianBlur(sigma=1.), iaa.LinearContrast((.5,1.5), per_channel=.5),
                iaa.Affine(scale={'x':(.5,2),'y':(.5,2)}), iaa.Affine(rotate=(-40,40)),
                iaa.Affine(shear=(-16,16)), iaa.PiecewiseAffine(scale=(.008,.03)),
                iaa.Affine(translate_percent={'x':(-.2,.2),'y':(-.2,.2)})], random_order=True)

def config():
    cfg = read(ROOT/'operations/focaltransnet_isic_20261114/config.json')
    cfg.update(experiment='FocalTransNet_HAM_FRESH100_20261114', dataset='HAM10000',
               split_source=str(SPLIT), split_counts={'train':6983,'val':1010,'test':2020},
               output_root=str(RUN),
               author_script_deviation='No HAM recipe in author repository; ISIC model/training adapted to frozen HAM split; validation selects checkpoint.',
               final_test={'cases':2020, 'modes':['none','mirror4'], 'prediction':'argmax of averaged logits'})
    return cfg

def preflight():
    focal.seed_everything(1234)
    torch.set_num_threads(4)
    dataset = HamSplit('train', True)
    loader = DataLoader(dataset, batch_size=6, shuffle=True, num_workers=0)
    model = focal.build_model().cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5, betas=(.9,.999), weight_decay=.03)
    criterion = focal.OfficialIsicLoss()
    torch.cuda.reset_peak_memory_stats()
    steps=[]
    for image, mask, ids in loader:
        if len(image) != 6:
            continue
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = criterion(model(image.cuda()), mask.cuda())
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite loss')
        loss.backward()
        grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not grads or not all(torch.isfinite(g).all().item() for g in grads):
            raise RuntimeError('missing or nonfinite gradients')
        optimizer.step()
        steps.append({'loss':float(loss.detach()),'image_ids':list(ids)})
        if len(steps)==3:
            break
    if len(steps)!=3:
        raise RuntimeError('preflight did not complete three batch6 steps')
    write(OPS/'preflight.json', {'passed':True,'steps':steps,'batch':6,
         'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
         'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
         'test_accessed':False,'weights_discarded':True,'split_sha256':focal.sha256(SPLIT)})
    print('HAM_PREFLIGHT_PASS',flush=True)

def train():
    focal.CONFIG_PATH = OPS/'config.json'
    focal.IsicSplit = HamSplit
    focal.train(RUN, torch.device('cuda:0'))

def test():
    if not (RUN/'train_complete.json').is_file() or not (RUN/'best.pt').is_file():
        raise RuntimeError('training completion or selected best checkpoint missing')
    from networks.Net import Net
    checkpoint=RUN/'best.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=focal.build_model()
    model.load_state_dict(state['state_dict'],strict=True)
    assert not torch.cuda.is_initialized()
    model.cuda().eval()
    ds=HamSplit('test',False)
    loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=0)
    expected=[r['image_id'] for r in ds.rows]
    out=RUN/'final_test'
    out.mkdir(exist_ok=False)
    results={}
    for mode in ('none','mirror4'):
        target=out/mode
        pred_dir=target/'predictions'
        pred_dir.mkdir(parents=True,exist_ok=False)
        totals=[0,0,0,0]; records=[]; seen=[]
        with torch.inference_mode():
            for image,mask,ids in loader:
                image=image.cuda(); mask=mask.cuda().bool(); logits=model(image)
                if mode=='mirror4':
                    logits=(logits+model(image.flip(-1)).flip(-1)+model(image.flip(-2)).flip(-2)+model(image.flip((-2,-1))).flip((-2,-1)))/4
                pred=logits.argmax(1).bool(); image_id=ids[0]
                counts=[int((pred & mask).sum()),int((pred & ~mask).sum()),int((~pred & mask).sum()),int((~pred & ~mask).sum())]
                totals=[a+b for a,b in zip(totals,counts)]; seen.append(image_id)
                tp,fp,fn,tn=counts
                records.append({'image_id':image_id,'tp':tp,'fp':fp,'fn':fn,'tn':tn,'dice':2*tp/max(2*tp+fp+fn,1)})
                np.save(pred_dir/f'{image_id}.npy',pred[0].cpu().numpy().astype(np.uint8))
        if seen != expected or len(seen)!=2020 or len(set(seen))!=2020:
            raise RuntimeError('HAM prediction IDs differ from frozen test split')
        with (target/'per_image_metrics.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        tp,fp,fn,tn=totals
        summary={'dataset':'HAM10000','model':'FocalTransNet','images':2020,'split':'test',
                 'seed':1234,'selected_epoch':state['epoch'],'tta':mode=='mirror4',
                 'checkpoint_sha256':focal.sha256(checkpoint),'split_sha256':focal.sha256(SPLIT),
                 'global':{'dice':2*tp/max(2*tp+fp+fn,1),'iou':tp/max(tp+fp+fn,1),
                           'tp':tp,'fp':fp,'fn':fn,'tn':tn},'test_used_for_checkpoint_selection':False}
        audit={'complete':True,'prediction_count':len(seen),'metric_count':len(records),
               'ids_match_frozen_test_split':True,'checkpoint_sha256':focal.sha256(checkpoint)}
        write(target/'summary.json',summary);write(target/'artifact_audit.json',audit)
        results[mode]=summary
        print('TEST_COMPLETE',mode,summary['global'],flush=True)
    write(RUN/'test_complete.json',{'complete':True,'results':results,'updated_at':time.time()})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['preflight','train','test','verify'])
    args=parser.parse_args()
    if args.mode=='verify':
        rows=split_rows(); print('HAM_SPLIT_VERIFIED',len(rows),focal.sha256(SPLIT))
        for split in ('train','val','test'):
            ds=HamSplit(split,False); x,y,identifier=ds[0]
            print(split,len(ds),tuple(x.shape),tuple(y.shape),identifier)
    elif args.mode=='preflight': preflight()
    elif args.mode=='train': train()
    else: test()

"""Isolated cache-to-vendor nnUNet dataset adapter; train/val only.

Images are RGB /255 (nnUNet RGBTo01); masks thresholded at 127.5 as
project split convention. The on-disk cache is already resized to 256.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor

ROOT=Path('/home/PuMengYu/experiments/skin_segmentation/data/processed')
CONFIG={
 'HAM10000': (ROOT/'ham10000_group42', ROOT/'ham10000_group42_dedup_20261023/split.csv'),
 'ISIC2018': (ROOT/'isic2018_256_dedup_seed42_20261031', ROOT/'isic2018_256_dedup_seed42_20261031/split.csv'),
}

class SkinCases(nnUNetDataset):
    def __init__(self, name, split):
        if split not in ('train','val'):
            raise ValueError('training adapter only permits train or val')
        root,csv=CONFIG[name]
        df=pd.read_csv(csv)
        if df.image_id.duplicated().any() or df.cache_index.duplicated().any():
            raise ValueError('split IDs or cache indices are not unique')
        df=df[df.split==split]
        self.dataset=dict(zip(df.image_id.astype(str),df.cache_index.astype(int)))
        if not self.dataset: raise ValueError('empty split')
        self.images=np.load(root/'images.npy',mmap_mode='r')
        self.masks=np.load(root/'masks.npy',mmap_mode='r')
    def load_case(self,key):
        i=self.dataset[key]
        image=np.asarray(self.images[i],dtype=np.float32).transpose(2,0,1)[:,None]/255.
        seg=(np.asarray(self.masks[i])>127.5).astype(np.int16)[None,None]
        loc=DefaultPreprocessor._sample_foreground_locations(seg,[1])[1]
        if len(loc)==0: loc=np.empty((0,4),dtype=np.int64)
        return image,seg,{'class_locations':{1:loc}}

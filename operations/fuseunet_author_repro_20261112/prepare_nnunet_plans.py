from pathlib import Path
import json
import numpy as np
import pandas as pd
import SimpleITK as sitk
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
OUT=ROOT/'operations/fuseunet_author_repro_20261112'
RAW=OUT/'nnunet_raw'
for dataset_id,name,cache_name in [(901,'HAM10000','ham10000_group42'),(902,'ISIC2018','isic2018_256_dedup_seed42_20261031')]:
    cache=ROOT/'data/processed'/cache_name
    df=pd.read_csv(cache/'split.csv'); train=df[df.split=='train']
    ds=RAW/f'Dataset{dataset_id:03d}_{name}'; images_dir=ds/'imagesTr'; labels_dir=ds/'labelsTr'
    images_dir.mkdir(parents=True,exist_ok=True); labels_dir.mkdir(parents=True,exist_ok=True)
    imgs=np.load(cache/'images.npy',mmap_mode='r'); masks=np.load(cache/'masks.npy',mmap_mode='r')
    for row in train.itertuples(index=False):
        stem=str(row.image_id); idx=int(row.cache_index); arr=np.asarray(imgs[idx],dtype=np.uint8)
        for c in range(3):
            im=sitk.GetImageFromArray(arr[:,:,c].astype(np.float32)); im.SetSpacing((1.,1.))
            sitk.WriteImage(im,str(images_dir/f'{stem}_{c:04d}.nii.gz'),True)
        seg=(np.asarray(masks[idx])>127.5).astype(np.uint8)
        lab=sitk.GetImageFromArray(seg); lab.SetSpacing((1.,1.))
        sitk.WriteImage(lab,str(labels_dir/f'{stem}.nii.gz'),True)
    obj={'channel_names':{'0':'R','1':'G','2':'B'},'labels':{'background':0,'lesion':1},'numTraining':int(len(train)),'file_ending':'.nii.gz','name':name,'description':'Train split only, cached 256x256 RGB channels, 2D; unit pixel spacing for nnU-Net planning.'}
    (ds/'dataset.json').write_text(json.dumps(obj,indent=2)+'\n')
    print(f'{name}: corrected 2D modality layout for {len(train)} training cases',flush=True)

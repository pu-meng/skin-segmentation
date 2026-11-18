import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from fuse_skin_adapter import SkinCases, CONFIG
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from types import SimpleNamespace

def test_real_train_val():
    for name in CONFIG:
        tr, val = SkinCases(name,'train'), SkinCases(name,'val')
        assert not set(tr.keys()) & set(val.keys())
        x,y,p=tr.load_case(next(iter(tr.keys())))
        assert x.shape == (3,1,256,256) and y.shape == (1,1,256,256)
        assert np.isfinite(x).all() and set(np.unique(y)) <= {0,1}
        coords=p['class_locations'][1]
        assert coords.shape[1] == 4
        if len(coords): assert np.all(y[coords[:,0],coords[:,1],coords[:,2],coords[:,3]]==1)
        labels=SimpleNamespace(all_labels=[0,1],has_ignore_label=False)
        dl=nnUNetDataLoader2D(tr,2,(256,256),(256,256),labels,oversample_foreground_percent=.33)
        batch=next(dl)
        assert batch['data'].shape==(2,3,256,256)
        assert batch['seg'].shape==(2,1,256,256)

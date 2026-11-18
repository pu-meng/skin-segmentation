import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from common import environment
environment('')
import torch
from types import SimpleNamespace
from tta import SubsetPredictor
from test_views import Toy
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

def main():
    torch.set_num_threads(1); torch.manual_seed(17)
    x=torch.randn(1,7,8,9)
    a=SubsetPredictor(views=8,device=torch.device('cpu'),use_gaussian=True,tile_step_size=.5)
    b=nnUNetPredictor(device=torch.device('cpu'),use_gaussian=True,tile_step_size=.5)
    for p in (a,b):
        p.configuration_manager=SimpleNamespace(patch_size=[5,6,7])
        p.network=Toy(); p.label_manager=SimpleNamespace(num_segmentation_heads=3)
        p.allowed_mirroring_axes=(0,1,2)
    y=a.predict_sliding_window_return_logits(x); z=b.predict_sliding_window_return_logits(x)
    torch.testing.assert_close(y,z,rtol=0,atol=0)
    assert y.shape==(3,7,8,9)
    assert not torch.cuda.is_initialized()
    print('PASS Gaussian step0.5 window batch1 full sliding + reverse padding: original 8-view max_abs='+str(float((y-z).abs().max())))
if __name__=='__main__': main()

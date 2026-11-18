"""Independent CPU integration test, leaving active-arm source hashes intact."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util
from pathlib import Path
import sys
import unittest
P=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(P))
sys.path.insert(0,str(P/'source/comparison'))
sys.path.insert(0,str(Path(__file__).parent))
import torch
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t')
torch.set_num_threads(1)

class UnetV2Test(unittest.TestCase):
    def test_unet_v2_has_original_generator_and_new_injection(self):
        path=P/'source/bottleneck_mamba/experiment_unet_v2.py'
        self.assertTrue(path.is_file(),'UNet V2 runner not implemented')
        spec=importlib.util.spec_from_file_location('unet_v2_test_target',path)
        m=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        self.assertEqual(m.ARMS['unet_v2']['base'],'unet2d')
        self.assertEqual(m.ARMS['unet_v2']['tokens'],256)
        model=m.build_model('unet_v2',pretrained=False)
        bottleneck=m.bottleneck_of(model,'unet_v2')
        self.assertEqual(type(bottleneck.mixer).__name__,'NoDeltaInjectionMamba')
        self.assertTrue(hasattr(bottleneck.mixer,'dt_proj'))
        self.assertFalse(hasattr(bottleneck.mixer,'parameter_generator'))
        x=torch.randn(2,3,32,32)
        y=model(x)
        self.assertEqual(tuple(y.shape),(2,1,32,32))
        y.mean().backward()
        self.assertEqual([n for n,p in model.named_parameters() if p.grad is None],[])
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertFalse(torch.cuda.is_initialized())

if __name__=='__main__':unittest.main(verbosity=2)

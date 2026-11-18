"""Approved model builders preserve audited outputs and support fresh strict reload."""
import io
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys
import unittest
P=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(P),str(P/'source/comparison'),str(Path(__file__).parent)]
import torch
from experiment import ARMS, build_model
from audit_encoders import classifier,Encoder,SharedUNet
from final_test import configure,pipeline

torch.set_num_threads(2)


class Integration(unittest.TestCase):
    def test_approved_models_match_audited_adapter_and_test_reconstruction(self):
        configure()
        for name,params in [('lowformer_e1',12954917),('cpubone_b0_bftraining',9137041)]:
            with self.subTest(name=name):
                m=build_model(name,pretrained=True).eval()
                self.assertEqual(sum(p.numel() for p in m.parameters()),params)
                self.assertTrue(m.encoder.loading_audit['full_classifier_strict_load'])
                ref,channels,_=classifier(name)
                reference=SharedUNet(Encoder(ref,name),channels).eval()
                x=torch.rand(1,3,256,256)
                with torch.no_grad():
                    expected=m(x)
                    torch.testing.assert_close(expected,reference(x))
                saved=io.BytesIO();torch.save(m.state_dict(),saved);saved.seek(0)
                clone=pipeline.build_model(name).eval()
                clone.load_state_dict(torch.load(saved,map_location='cpu',weights_only=True),strict=True)
                with torch.no_grad():torch.testing.assert_close(expected,clone(x))
                self.assertEqual(tuple(expected.shape),(1,1,256,256))
                self.assertIn(name,pipeline.ARMS)
                del m,reference,ref,clone,saved

    def test_unapproved_candidate_not_enabled(self):
        with self.assertRaises(ValueError):build_model('visionhope_tiny',pretrained=False)


if __name__=='__main__':unittest.main(verbosity=2)

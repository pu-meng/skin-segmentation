"""V3 formula, gradient, checkpoint and final-test builder contracts (CPU)."""
import io
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import sys
import unittest
P = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(P), str(P/'source/comparison'), str(Path(__file__).parent)]
import torch
from torch.nn import functional as F
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t')
from Models.TinyViM.mamba_bottleneck_v3 import ConvGateMamba
from experiment_v3 import build_model, bottleneck_of
import final_test_pipeline as pipeline

torch.set_num_threads(2)

class V3Contracts(unittest.TestCase):
    def test_formula_and_trainable_D_without_independent_skip(self):
        torch.manual_seed(42)
        m = ConvGateMamba(8, d_state=3)
        x = torch.randn(2, 6, 8, requires_grad=True)
        f,z = m.in_proj(x).transpose(1,2).chunk(2,1)
        t = F.silu(m.conv1d(f)[...,:6])
        dt = F.softplus(m.dt_proj(m.x_proj(t.transpose(1,2)))).transpose(1,2)
        u = m.spatial_conv(t.reshape(2,16,2,3)).flatten(2)
        gu = sum(torch.exp(dt*(-m.A_log[:,n].exp())[None,:,None]) for n in range(3))
        for d in (0.0, -0.7, 1.0):
            with torch.no_grad(): m.D.fill_(d)
            expected = m.out_proj(((gu*u + d*dt*t)*F.silu(z)).transpose(1,2))
            torch.testing.assert_close(m(x,(2,3)),expected)
        y = m(x,(2,3)); y.square().mean().backward()
        self.assertTrue(m.D.requires_grad)
        self.assertGreater(m.D.grad.abs().sum().item(),0)
        self.assertEqual([n for n,p in m.named_parameters() if p.grad is None],[])
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters()))
        self.assertEqual(m.x_proj.out_features,m.dt_rank)
        with self.assertRaises(ValueError): m(x,(3,3))

    def test_both_full_models_and_final_test_checkpoint_roundtrip(self):
        for arm in ('unet_v3','tinyvim_v3'):
            with self.subTest(arm=arm):
                model=build_model(arm,pretrained=False)
                observed=[]
                bottleneck_of(model,arm).register_forward_hook(lambda m,i,o: observed.append(tuple(i[0].shape)))
                x=torch.randn(1,3,256,256)
                model.eval()
                # TinyViM's official backbone scan requires CUDA. Test its new
                # bottleneck on CPU; real full-network checks run in preflight.
                feature=torch.randn(1,512,8,8) if arm=='tinyvim_v3' else x
                target=bottleneck_of(model,arm) if arm=='tinyvim_v3' else model
                y=target(feature)
                self.assertEqual(tuple(y.shape),(1,512,8,8) if arm=='tinyvim_v3' else (1,1,256,256))
                self.assertEqual(observed[0],(1,512,16,16) if arm=='unet_v3' else (1,512,8,8))
                y.mean().backward()
                self.assertEqual([n for n,p in target.named_parameters() if p.grad is None],[])
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in target.parameters()))
                saved=io.BytesIO(); torch.save(model.state_dict(),saved); saved.seek(0)
                clone=pipeline.build_model(arm).eval()
                clone.load_state_dict(torch.load(saved,map_location='cpu',weights_only=True),strict=True)
                clone_target=bottleneck_of(clone,arm) if arm=='tinyvim_v3' else clone
                with torch.no_grad(): torch.testing.assert_close(clone_target(feature),y.detach())
                self.assertIn(arm,pipeline.ARMS)
                self.assertFalse(pipeline.training_ready(pipeline.ARMS[arm][1]))
                del model,clone,y,saved
        self.assertFalse(torch.cuda.is_initialized())

if __name__=='__main__': unittest.main(verbosity=2)

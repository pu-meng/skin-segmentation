"""Separable projection identity initialization, V2 equivalence and both model builders."""
import io
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import sys
import unittest
P = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(P), str(P/'source/comparison'), str(Path(__file__).parent)]
import torch
from torch import nn
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t')
from Models.TinyViM.mamba_bottleneck_v2_ds_proj import SpatialProjectionMambaV2
from Models.TinyViM.mamba_bottleneck_v2 import NoDeltaInjectionMamba
from experiment_v2_ds_proj import build_model, bottleneck_of
import final_test_pipeline as pipeline

torch.set_num_threads(2)


class V2SeparableProjectionContracts(unittest.TestCase):
    def test_identity_projection_preserves_V2_forward_and_gradients(self):
        torch.manual_seed(44)
        new = SpatialProjectionMambaV2(8, d_state=3)
        torch.manual_seed(44)
        reference = NoDeltaInjectionMamba(8, d_state=3, expand=2)
        self.assertEqual(new.d_inner, 16)
        self.assertEqual(tuple(new.A_log.shape), (16, 3))
        for name, p in reference.named_parameters():
            if not name.startswith(('in_proj.', 'out_proj.')):
                torch.testing.assert_close(dict(new.named_parameters())[name], p)
        x = torch.randn(2, 6, 8, requires_grad=True)
        xr = x.detach().clone().requires_grad_()
        y, yr = new(x, (2, 3)), reference(xr)
        torch.testing.assert_close(y, yr)
        y.square().mean().backward()
        yr.square().mean().backward()
        torch.testing.assert_close(x.grad, xr.grad)
        for name, p in new.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
            if not name.startswith(('in_proj.', 'out_proj.')):
                torch.testing.assert_close(p.grad, dict(reference.named_parameters())[name].grad)
        for name in ('in_proj', 'out_proj'):
            projection = getattr(new, name)
            torch.testing.assert_close(projection[1].weight.grad[:, :, 0, 0],
                                       getattr(reference, name).weight.grad)
            self.assertGreater(projection[0].weight.grad.abs().sum().item(), 0)
        with self.assertRaises(ValueError):
            new(x, (3, 3))

    def test_both_models_and_final_test_strict_reload(self):
        for arm in ('unet_v2_ds_proj', 'tinyvim_v2_ds_proj'):
            with self.subTest(arm=arm):
                torch.manual_seed(42)
                model = build_model(arm, pretrained=False).eval()
                module = bottleneck_of(model, arm)
                self.assertEqual(sum(p.numel() for p in module.parameters()), 1709568)
                observed = []
                module.register_forward_hook(lambda m,i,o: observed.append(tuple(i[0].shape)))
                # TinyViM backbone's official scan needs CUDA; full-network
                # validation is a mandatory real-GPU launch preflight.
                tiny = arm.startswith('tinyvim')
                x = torch.randn(1,512,8,8) if tiny else torch.randn(1,3,256,256)
                target = module if tiny else model
                y = target(x)
                self.assertEqual(tuple(y.shape), (1,512,8,8) if tiny else (1,1,256,256))
                self.assertEqual(observed[0], (1,512,8,8) if tiny else (1,512,16,16))
                y.square().mean().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in target.parameters()))
                saved = io.BytesIO()
                torch.save(model.state_dict(), saved)
                saved.seek(0)
                clone = pipeline.build_model(arm).eval()
                clone.load_state_dict(torch.load(saved, map_location='cpu', weights_only=True), strict=True)
                with torch.no_grad():
                    result = bottleneck_of(clone,arm)(x) if tiny else clone(x)
                torch.testing.assert_close(result, y.detach())
                self.assertIn(arm, pipeline.ARMS)
                del model, module, clone, target, saved, y
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main(verbosity=2)

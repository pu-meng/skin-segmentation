"""V2 semantics, shared input/gate gradients, both builders and reload contract."""
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
from Models.TinyViM.mamba_bottleneck_v2_no_proj import NoProjectionMambaV2
from Models.TinyViM.mamba_bottleneck_v2 import NoDeltaInjectionMamba
from experiment_v2_no_proj import build_model, bottleneck_of
import final_test_pipeline as pipeline

torch.set_num_threads(2)


class DuplicateInput(nn.Module):
    def forward(self, x):
        return torch.cat((x, x), dim=-1)


class V2NoProjectionContracts(unittest.TestCase):
    def test_original_v2_equivalence_and_gradients_at_width_C0(self):
        torch.manual_seed(44)
        new = NoProjectionMambaV2(8, d_state=3)
        reference = NoDeltaInjectionMamba(8, d_state=3, expand=1)
        reference.in_proj = DuplicateInput()
        reference.out_proj = nn.Identity()
        reference.load_state_dict(new.state_dict(), strict=True)
        self.assertEqual(new.d_inner, new.d_model)
        self.assertFalse(hasattr(new, 'in_proj'))
        self.assertFalse(hasattr(new, 'out_proj'))
        self.assertEqual(tuple(new.A_log.shape), (8, 3))
        self.assertEqual(new.x_proj.out_features, new.dt_rank+6)
        x = torch.randn(2, 9, 8, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_()
        y, y_ref = new(x), reference(x_ref)
        torch.testing.assert_close(y, y_ref)
        y.square().mean().backward()
        y_ref.square().mean().backward()
        torch.testing.assert_close(x.grad, x_ref.grad)
        reference_parameters = dict(reference.named_parameters())
        for name, p in new.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
            torch.testing.assert_close(p.grad, reference_parameters[name].grad)

    def test_both_models_and_final_test_strict_reload(self):
        for arm in ('unet_v2_no_proj', 'tinyvim_v2_no_proj'):
            with self.subTest(arm=arm):
                torch.manual_seed(42)
                model = build_model(arm, pretrained=False).eval()
                module = bottleneck_of(model, arm)
                self.assertEqual(sum(p.numel() for p in module.parameters()), 61952)
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

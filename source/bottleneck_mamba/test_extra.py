"""CPU contracts for the independently authorized additional arms."""
import importlib.util
import os
from pathlib import Path
import sys
import unittest

os.environ['CUDA_VISIBLE_DEVICES'] = ''
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'source/comparison'))
import torch
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t')
torch.set_num_threads(1)


def load_v2():
    path = PROJECT / 'Models/TinyViM/mamba_bottleneck_v2.py'
    spec = importlib.util.spec_from_file_location('v2_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AdditionalContracts(unittest.TestCase):
    def test_additional_models_have_separate_correct_bottlenecks(self):
        path = PROJECT / 'source/bottleneck_mamba/experiment_extra.py'
        self.assertTrue(path.is_file(), 'Additional integration runner is missing')
        sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location('extra_under_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        tiny = module.build_model('tinyvim_v2', pretrained=False)
        self.assertTrue(hasattr(tiny.align[-1][1].mixer, 'dt_proj'))
        self.assertFalse(hasattr(tiny.align[-1][1].mixer, 'parameter_generator'))
        unet = module.build_model('unet_v1', pretrained=False)
        self.assertEqual(len(unet.encoders), 5)
        self.assertTrue(hasattr(unet.encoders[-1][1].mixer, 'parameter_generator'))
        self.assertFalse(hasattr(unet.encoders[-1][1].mixer, 'dt_proj'))
        x = torch.randn(2, 3, 32, 32)
        y = unet(x)
        self.assertEqual(tuple(y.shape), (2, 1, 32, 32))
        y.mean().backward()
        self.assertEqual([n for n,p in unet.named_parameters() if p.grad is None], [])
        self.assertFalse(torch.cuda.is_initialized())

    def test_v2_changes_only_injection_not_transition_or_parameter_generator(self):
        path = PROJECT / 'Models/TinyViM/mamba_bottleneck_v2.py'
        self.assertTrue(path.is_file(), 'V2 implementation has not been added')
        module = load_v2()
        u = torch.randn(2, 4, 5, requires_grad=True)
        delta = torch.full_like(u, 0.1, requires_grad=True)
        A = -torch.ones(4, 3)
        B = torch.randn(2, 3, 5, requires_grad=True)
        C = torch.randn(2, 3, 5, requires_grad=True)
        actual = module.scan_without_delta_injection(u, delta, A, B, C)
        # First token starts at zero state: its output must not depend on delta.
        expected_first = (B[:, :, 0] * C[:, :, 0]).sum(-1)[:, None] * u[:, :, 0]
        torch.testing.assert_close(actual[:, :, 0], expected_first)
        h0 = B[:, None, :, 0] * u[:, :, None, 0]
        h1 = torch.exp(delta[:, :, None, 1] * A[None]) * h0 + B[:, None, :, 1] * u[:, :, None, 1]
        expected_second = (h1 * C[:, None, :, 1]).sum(-1)
        torch.testing.assert_close(actual[:, :, 1], expected_second)
        actual.square().mean().backward()
        for t in [u, delta, B, C]:
            self.assertIsNotNone(t.grad)
            self.assertTrue(torch.isfinite(t.grad).all())
        torch.manual_seed(42)
        model = module.NoDeltaInjectionMamba(d_model=8, d_state=3)
        self.assertTrue(hasattr(model, 'x_proj'))
        self.assertTrue(hasattr(model, 'dt_proj'))
        self.assertFalse(hasattr(model, 'parameter_generator'))
        x = torch.randn(2, 5, 8, requires_grad=True)
        y = model(x)
        self.assertEqual(x.shape, y.shape)
        y.square().mean().backward()
        self.assertEqual([n for n,p in model.named_parameters() if p.grad is None], [])
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        clone = module.NoDeltaInjectionMamba(d_model=8, d_state=3)
        clone.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(clone(x.detach()), y.detach())
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main(verbosity=2)

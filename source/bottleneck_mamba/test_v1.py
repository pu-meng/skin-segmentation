"""CPU contract test for the separately implemented bottleneck V1."""
import importlib.util
import os
from pathlib import Path
import sys
import unittest

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'source/comparison'))

import torch
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t')
torch.set_num_threads(1)


class V1Contract(unittest.TestCase):
    def test_integration_only_modifies_deepest_alignment(self):
        path = PROJECT / 'source/bottleneck_mamba/experiment_v1.py'
        self.assertTrue(path.is_file(), 'V1 integration runner is not present yet')
        sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location('experiment_under_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        audit = module.verify_v1_runtime()
        self.assertIn('checked_files', audit)
        self.assertTrue(all('draw_model_structure.py' in x['path']
                            for x in audit['unrelated_plotting_drift']))
        model = module.build_experiment_model(pretrained=False)
        self.assertEqual(len(model.align), 4)
        self.assertIsInstance(model.align[-1], torch.nn.Sequential)
        self.assertEqual(model.align[-1][0].in_channels, 224)
        self.assertEqual(model.align[-1][0].out_channels, 512)
        self.assertEqual(model.align[-1][1].mixer.d_model, 512)
        self.assertEqual(model.align[-1][1].mixer.d_state, 16)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_direct_nonlinear_generator_forward_backward_and_state(self):
        path = PROJECT / 'Models/TinyViM/mamba_bottleneck_v1.py'
        self.assertTrue(path.is_file(), 'V1 implementation is not present yet')
        spec = importlib.util.spec_from_file_location('v1_under_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        torch.manual_seed(42)
        model = module.DirectParameterMamba(d_model=8, d_state=3, expand=2)
        self.assertFalse(hasattr(model, 'x_proj'))
        self.assertFalse(hasattr(model, 'dt_proj'))
        u = torch.randn(2, model.d_inner, 7)
        raw_dt, B, C = model.generate_parameters(u)
        self.assertEqual(tuple(raw_dt.shape), (2, 16, 7))
        self.assertEqual(tuple(B.shape), (2, 3, 7))
        self.assertEqual(tuple(C.shape), (2, 3, 7))
        self.assertTrue((torch.nn.functional.softplus(raw_dt) > 0).all())
        self.assertTrue(any(isinstance(m, torch.nn.SiLU)
                            for m in model.parameter_generator.modules()))
        x = torch.randn(2, 7, 8, requires_grad=True)
        y = model(x)
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        missing = [n for n, p in model.named_parameters() if p.grad is None]
        self.assertEqual(missing, [])
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        other = module.DirectParameterMamba(d_model=8, d_state=3, expand=2)
        other.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(other(x.detach()), y.detach())
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main(verbosity=2)

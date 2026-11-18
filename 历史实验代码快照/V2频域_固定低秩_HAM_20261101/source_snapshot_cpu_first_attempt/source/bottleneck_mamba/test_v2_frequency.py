"""CPU-only synthetic contracts for the original V2 parallel frequency branch."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import sys
import unittest
P = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(P), str(P/'source/comparison'), str(Path(__file__).parent)]
import torch
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t')
torch.set_num_threads(1)


class FrequencyContracts(unittest.TestCase):
    def test_fixed_shape(self):
        self.assertTrue((P/'Models/TinyViM/mamba_bottleneck_v2_frequency.py').is_file(),
                        'Frequency wrapper is not implemented')
        from Models.TinyViM.mamba_bottleneck_v2_frequency import BottleneckMambaV2Frequency
        model = BottleneckMambaV2Frequency(channels=8, height=3, width=4, d_state=3)
        x = torch.randn(2, 8, 3, 4)
        y = model(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(y).all())
        self.assertFalse(torch.cuda.is_initialized())


class ExtendedFrequencyContracts(unittest.TestCase):
    @staticmethod
    def make(mode='fixed', **kwargs):
        from Models.TinyViM.mamba_bottleneck_v2_frequency import BottleneckMambaV2Frequency
        return BottleneckMambaV2Frequency(channels=kwargs.pop('channels', 8),
            height=kwargs.pop('height', 3), width=kwargs.pop('width', 4),
            d_state=kwargs.pop('d_state', 3), mode=mode, **kwargs)

    def test_all_parameter_gradients_real_512_8_8(self):
        # A nonzero norm per tensor, not a claim that every individual element is nonzero.
        GRADIENTS.clear()
        for mode in ('fixed', 'dynamic'):
            with self.subTest(mode=mode):
                torch.manual_seed(2026)
                model = self.make(mode, channels=512, height=8, width=8, d_state=16)
                x = torch.randn(2, 512, 8, 8, requires_grad=True)
                y = model(x)
                self.assertEqual(y.shape, x.shape)
                self.assertTrue(torch.isfinite(y).all())
                ((y * torch.randn_like(y)).mean() + y.square().mean()).backward()
                norms = {}
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    norm = float(parameter.grad.double().norm())
                    self.assertGreater(norm, 0, name)
                    norms[name] = norm
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertGreater(float(x.grad.norm()), 0)
                self.assertIsInstance(model.v2, __import__('Models.TinyViM.mamba_bottleneck_v2',
                    fromlist=['BottleneckMambaV2']).BottleneckMambaV2)
                GRADIENTS[mode] = {'shape': list(y.shape), 'parameter_tensor_grad_norms': norms,
                                   'input_grad_norm': float(x.grad.norm()),
                                   'parameters': sum(p.numel() for p in model.parameters())}
        self.assertFalse(torch.cuda.is_initialized())

    def test_alpha_zero_bitwise_original_v2(self):
        from Models.TinyViM.mamba_bottleneck_v2 import BottleneckMambaV2
        for mode in ('fixed', 'dynamic'):
            model = self.make(mode).eval()
            reference = BottleneckMambaV2(channels=8, d_state=3, seed=44).eval()
            reference.load_state_dict(model.v2.state_dict(), strict=True)
            x = torch.randn(2, 8, 3, 4)
            with torch.no_grad():
                model.alpha.zero_()
                self.assertTrue(torch.equal(model(x), reference(x)))

    def test_parallel_formula_and_full_frequency_not_difference(self):
        for mode in ('fixed', 'dynamic'):
            model = self.make(mode)
            x = torch.randn(2, 8, 3, 4)
            with torch.no_grad():
                gain = model.frequency_gain(x)
                expected_frequency = torch.fft.irfft2(gain * torch.fft.rfft2(x.float(), norm='ortho'),
                    s=(3, 4), norm='ortho')
                torch.testing.assert_close(model.frequency_branch(x), expected_frequency, rtol=0, atol=0)
                torch.testing.assert_close(model(x), model.v2(x) + model.alpha * expected_frequency,
                                           rtol=0, atol=0)
                self.assertGreater(float((model.frequency_branch(x) - x).abs().max()), 0)
                self.assertGreater(float(model.frequency_branch(x).norm()), float(x.norm()) * 0.9)

    def test_fft_odd_even_roundtrip(self):
        for h, w in ((3, 5), (4, 6), (3, 6), (4, 5), (1, 1)):
            x = torch.randn(2, 8, h, w)
            restored = torch.fft.irfft2(torch.fft.rfft2(x, norm='ortho'), s=(h, w), norm='ortho')
            torch.testing.assert_close(restored, x, rtol=1e-5, atol=1e-6)
            for mode in ('fixed', 'dynamic'):
                model = self.make(mode, height=h, width=w)
                self.assertEqual(model(x).shape, x.shape)

    def test_gain_boundary_symmetry_and_spectrum_validity(self):
        for h, w in ((3, 5), (4, 6), (3, 6), (4, 5)):
            for mode in ('fixed', 'dynamic'):
                model = self.make(mode, height=h, width=w)
                x = torch.randn(2, 8, h, w)
                g = model.frequency_gain(x)
                reverse = (-torch.arange(h)) % h
                for col in ([0, w//2] if w % 2 == 0 else [0]):
                    self.assertTrue(torch.equal(g[..., col], g.index_select(-2, reverse)[..., col]))
                spectrum = g * torch.fft.rfft2(x, norm='ortho')
                restored_spectrum = torch.fft.rfft2(model.frequency_branch(x), norm='ortho')
                torch.testing.assert_close(spectrum, restored_spectrum, rtol=1e-5, atol=1e-6)
                self.assertTrue((g >= 0.9).all() and (g <= 1.1).all())

    def test_gain_sample_sharing_versus_dynamic_difference(self):
        x = torch.stack((torch.zeros(8, 3, 4), torch.full((8, 3, 4), 5.0)))
        for mode in ('fixed', 'dynamic'):
            model = self.make(mode)
            g = model.frequency_gain(x)
            self.assertEqual(torch.equal(g[0], g[1]), mode == 'fixed')
            if mode == 'fixed':
                self.assertEqual(tuple(model.U.shape), (8, 4))
                self.assertEqual(tuple(model.V.shape), (4, 9))
            else:
                self.assertEqual(model.generator[0].out_features, 32)
                self.assertIsInstance(model.generator[1], torch.nn.GELU)

    def test_zero_input_finite(self):
        for mode in ('fixed', 'dynamic'):
            model = self.make(mode)
            x = torch.zeros(2, 8, 3, 4)
            self.assertTrue(torch.isfinite(model(x)).all())
            self.assertTrue(torch.equal(model.frequency_branch(x), x))

    def test_strict_state_dict_roundtrip(self):
        import io
        for mode in ('fixed', 'dynamic'):
            model, clone = self.make(mode).eval(), self.make(mode).eval()
            x = torch.randn(2, 8, 3, 4)
            stream = io.BytesIO()
            torch.save(model.state_dict(), stream)
            stream.seek(0)
            clone.load_state_dict(torch.load(stream, map_location='cpu', weights_only=True), strict=True)
            self.assertTrue(torch.equal(model(x), clone(x)))
            bad = dict(model.state_dict())
            bad.pop('alpha')
            with self.assertRaises(RuntimeError):
                clone.load_state_dict(bad, strict=True)

    def test_invalid_dimensions_and_input_dtype_rejected(self):
        model = self.make()
        for shape in ((2, 8, 4, 4), (2, 7, 3, 4), (2, 8, 3), (0, 8, 3, 4)):
            with self.assertRaisesRegex(ValueError, 'Expected nonempty'):
                model(torch.randn(*shape))
        with self.assertRaises(TypeError):
            model(torch.zeros(2, 8, 3, 4, dtype=torch.int64))
        for kwargs in ({'height': 0}, {'rank': 0}, {'width': 2.5}, {'epsilon': 1.0}, {'mode': 'other'}):
            with self.assertRaises(ValueError):
                self.make(**kwargs)
        with self.assertRaises(TypeError):
            self.make().double()(torch.randn(2, 8, 3, 4, dtype=torch.float64))

    def test_branch_fp32_under_cpu_autocast(self):
        for mode in ('fixed', 'dynamic'):
            model = self.make(mode)
            x = torch.randn(2, 8, 3, 4)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                self.assertEqual(model.frequency_gain(x).dtype, torch.float32)
                self.assertEqual(model.frequency_branch(x).dtype, torch.float32)

    def test_initialization_nonzero_small_and_defaults(self):
        for mode in ('fixed', 'dynamic'):
            model = self.make(mode)
            self.assertEqual(model.rank, 4)
            self.assertEqual(model.epsilon, 0.1)
            self.assertAlmostEqual(float(model.alpha.detach()), 0.01)
            for name, p in model.named_parameters():
                if not name.startswith('v2.'):
                    self.assertGreater(float(p.detach().norm()), 0, name)
                    self.assertTrue(torch.isfinite(p).all(), name)
                    self.assertLess(float(p.detach().abs().max()), 0.2, name)


GRADIENTS = {}

if __name__ == '__main__':
    import argparse
    import json
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument('--checks', type=Path)
    args, remaining = parser.parse_known_args()
    class RecordedResult(unittest.TextTestResult):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.test_ids = []

        def startTest(self, test):
            self.test_ids.append(test.id())
            super().startTest(test)

    torch.manual_seed(2026)
    started = time.monotonic()
    runner = unittest.TextTestRunner(verbosity=2, resultclass=RecordedResult)
    program = unittest.main(argv=[sys.argv[0], *remaining], testRunner=runner, exit=False)
    result = program.result
    record = {'passed': result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped),
              'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
              'skipped': len(result.skipped), 'successful': result.wasSuccessful(),
              'test_ids': result.test_ids, 'seed': 2026, 'argv': sys.argv,
              'device': 'cpu', 'threads': torch.get_num_threads(),
              'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
              'cuda_initialized': torch.cuda.is_initialized(), 'python': sys.version,
              'torch': torch.__version__, 'elapsed_seconds': time.monotonic() - started,
              'real_interface_gradient_checks': GRADIENTS,
              'failure_details': [{'test': str(t), 'traceback': e} for t,e in result.failures + result.errors]}
    if args.checks:
        args.checks.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ('real_interface_gradient_checks', 'failure_details')},
                     ensure_ascii=False), flush=True)
    sys.exit(0 if result.wasSuccessful() and not record['cuda_initialized'] else 1)

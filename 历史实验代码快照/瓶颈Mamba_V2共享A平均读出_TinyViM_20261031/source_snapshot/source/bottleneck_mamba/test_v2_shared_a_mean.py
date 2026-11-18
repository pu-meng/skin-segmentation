"""Equation, initialization, gradients, integration and reload contracts."""
import argparse
import io
import json
from pathlib import Path
import sys
import unittest
P = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(P), str(P/'source/comparison'), str(Path(__file__).parent)]
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t')
import torch
from Models.TinyViM.mamba_bottleneck_v2_shared_a_mean import shared_a_mean_scan, SharedAMeanMambaV2
from Models.TinyViM.mamba_bottleneck_v2 import NoDeltaInjectionMamba
from experiment_v2_shared_a_mean import build_model, bottleneck_of

DEVICE = 'cpu'
RECORDS = []


def reference(u, delta, A, B):
    state = torch.zeros(u.shape[0], u.shape[1], A.shape[1], device=u.device)
    result = []
    for t in range(u.shape[2]):
        state = torch.exp(delta[:, :, t, None] * A[None]) * state + B[:, None, :, t] * u[:, :, t, None]
        result.append(state.mean(-1))
    return torch.stack(result, -1)


class CandidateContracts(unittest.TestCase):
    def test_equations_and_shared_parameter_gradients(self):
        for length, small in [(7,False), (64,False), (256,False), (7,True)]:
            torch.manual_seed(44)
            u = torch.randn(2, 5, length, device=DEVICE)
            delta = torch.rand_like(u) * .07
            if small:
                delta.fill_(1e-25)
                delta[:, :, 0] = 0
            values = [u, delta, -torch.rand(1, 3, device=DEVICE), torch.randn(2, 3, length, device=DEVICE)]
            left = [v.detach().clone().requires_grad_() for v in values]
            right = [v.detach().clone().requires_grad_() for v in values]
            actual, expected = shared_a_mean_scan(*left), reference(*right)
            torch.testing.assert_close(actual, expected, rtol=3e-4, atol=5e-4)
            weights = torch.randn_like(actual)
            (actual*weights).sum().backward()
            (expected*weights).sum().backward()
            for name,a,b in zip(('u','delta','A','B'),left,right):
                self.assertIsNotNone(a.grad, name)
                self.assertTrue(torch.isfinite(a.grad).all(), name)
                torch.testing.assert_close(a.grad, b.grad, rtol=3e-3, atol=7e-4)
            self.assertEqual(tuple(left[2].grad.shape), (1,3))
            RECORDS.append({'test':'scan_equivalence','device':DEVICE,'length':length,'tiny_zero_delta':small,
                            'max_abs':float((actual-expected).abs().max().detach())})

    def test_parent_initialization_and_removed_C(self):
        torch.manual_seed(44)
        original = NoDeltaInjectionMamba(512, d_state=16)
        torch.manual_seed(44)
        candidate = SharedAMeanMambaV2(512, d_state=16)
        self.assertEqual(candidate.x_proj.out_features, 32+16)
        self.assertEqual(tuple(candidate.A_log.shape),(1,16))
        self.assertEqual(sum(p.numel() for p in candidate.parameters()),1661968)
        for name,param in candidate.named_parameters():
            expected = dict(original.named_parameters())[name]
            if name == 'x_proj.weight':expected=expected[:48]
            if name == 'A_log':expected=expected[:1]
            torch.testing.assert_close(param, expected)

    def test_builders_update_save_reload_and_shapes(self):
        for arm in ('tinyvim_v2_shared_a_mean', 'unet_v2_shared_a_mean'):
            torch.manual_seed(42)
            model = build_model(arm, pretrained=False).to(DEVICE).eval()
            module = bottleneck_of(model, arm)
            self.assertEqual(sum(p.numel() for p in module.parameters()),1662992)
            tiny = arm.startswith('tinyvim')
            # Official TinyViM scan is CUDA-only; full TinyViM preflight is mandatory.
            target = module if tiny and DEVICE=='cpu' else model
            x = torch.randn(1,512,8,8,device=DEVICE) if target is module else torch.randn(1,3,256,256,device=DEVICE)
            optimizer = torch.optim.AdamW(target.parameters(),lr=.001,weight_decay=.01)
            before=module.mixer.A_log.detach().clone()
            out=target(x)
            self.assertEqual(tuple(out.shape),tuple(x.shape) if target is module else (1,1,256,256))
            out.square().mean().backward()
            for name,param in target.named_parameters():
                self.assertIsNotNone(param.grad,name)
                self.assertTrue(torch.isfinite(param.grad).all(),name)
            optimizer.step()
            self.assertFalse(torch.equal(before,module.mixer.A_log))
            with torch.no_grad():expected=target(x).clone()
            buffer=io.BytesIO();torch.save(model.state_dict(),buffer);buffer.seek(0)
            clone=build_model(arm,pretrained=False).to(DEVICE).eval()
            clone.load_state_dict(torch.load(buffer,map_location=DEVICE,weights_only=True),strict=True)
            with torch.no_grad():actual=bottleneck_of(clone,arm)(x) if target is module else clone(x)
            torch.testing.assert_close(actual,expected)
            RECORDS.append({'test':'integration_update_reload','arm':arm,'device':DEVICE,
                            'scope':'bottleneck only' if target is module else 'full model',
                            'model_parameters':sum(p.numel() for p in model.parameters()),
                            'bottleneck_parameters':1662992,'strict_reload':True,'shared_A_updated':True})
            del model,module,clone,target,optimizer,out,expected,actual,buffer


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args();DEVICE=args.device
    torch.set_num_threads(2)
    if DEVICE=='cuda' and not torch.cuda.is_available():raise RuntimeError('Real CUDA required; cannot skip GPU contract')
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CandidateContracts))
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps({'passed':result.wasSuccessful(),'tests_run':result.testsRun,
        'failures':len(result.failures),'errors':len(result.errors),'device':DEVICE,'checks':RECORDS},ensure_ascii=False,indent=2)+'\n')
    sys.exit(0 if result.wasSuccessful() else 1)

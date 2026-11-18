"""Contract for isolated, equation-preserving V2 CUDA reuse."""
from pathlib import Path
import sys
import unittest
import json
import os

PROJECT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(PROJECT),str(PROJECT/'source/comparison'),str(Path(__file__).parent)]
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t')
import torch
from Models.TinyViM.mamba_bottleneck_v2 import scan_without_delta_injection,BottleneckMambaV2

class Equivalence(unittest.TestCase):
    def test_outputs_gradients_and_state_dict_match(self):
        p=PROJECT/'Models/TinyViM/mamba_bottleneck_v2_cuda_pilot.py'
        self.assertTrue(p.exists(),'Independent CUDA-equivalent V2 module not implemented')
        from Models.TinyViM.mamba_bottleneck_v2_cuda_pilot import scan_v2_cuda_equivalent,BottleneckMambaV2CudaPilot
        torch.set_num_threads(2)
        reports=[]
        for device in ['cpu']+(['cuda:0'] if torch.cuda.is_available() else []):
            for length in [7,64,256]:
                torch.manual_seed(44)
                u=torch.randn(2,32,length,device=device)
                delta=torch.exp(torch.empty_like(u).uniform_(-9,-2))
                A=-torch.rand(32,16,device=device)*4-0.1
                B=torch.randn(2,16,length,device=device)
                C=torch.randn(2,16,length,device=device)
                left=[v.detach().clone().requires_grad_() for v in [u,delta,A,B,C]]
                right=[v.detach().clone().requires_grad_() for v in [u,delta,A,B,C]]
                y=scan_without_delta_injection(*left);z=scan_v2_cuda_equivalent(*right)
                torch.testing.assert_close(y,z,rtol=3e-4,atol=5e-4)
                weight=torch.randn_like(y)
                (y*weight).mean().backward();(z*weight).mean().backward()
                errors={}
                for name,a,b in zip(['u','delta','A','B','C'],left,right):
                    self.assertTrue(torch.isfinite(b.grad).all())
                    torch.testing.assert_close(a.grad,b.grad,rtol=3e-3,atol=5e-4)
                    errors[name]=float((a.grad-b.grad).abs().max())
                reports.append({'device':device,'length':length,'output_max_abs':float((y-z).abs().max()),'gradient_max_abs':errors})
        old=BottleneckMambaV2(512,16);new=BottleneckMambaV2CudaPilot(512,16)
        new.load_state_dict(old.state_dict(),strict=True)
        self.assertEqual(set(old.state_dict()),set(new.state_dict()))
        self.assertEqual(sum(p.numel() for p in old.parameters()),sum(p.numel() for p in new.parameters()))
        device='cuda:0' if torch.cuda.is_available() else 'cpu'
        old=old.to(device);new=new.to(device)
        a=torch.randn(1,512,4,4,device=device,requires_grad=True);b=a.detach().clone().requires_grad_()
        y=old(a);z=new(b);torch.testing.assert_close(y,z,rtol=5e-4,atol=5e-4)
        y.square().mean().backward();z.square().mean().backward()
        torch.testing.assert_close(a.grad,b.grad,rtol=3e-3,atol=5e-4)
        for (na,pa),(nb,pb) in zip(old.named_parameters(),new.named_parameters()):
            self.assertEqual(na,nb);self.assertIsNotNone(pb.grad)
            torch.testing.assert_close(pa.grad,pb.grad,rtol=3e-3,atol=5e-4)
        report={'status':'passed','scan_checks':reports,'strict_state_dict':True,'all_module_parameter_gradients':True,
                'parameters':sum(p.numel() for p in new.parameters()),'cuda_tested':torch.cuda.is_available()}
        if os.environ.get('CUDA_PILOT_TEST_REPORT'):
            Path(os.environ['CUDA_PILOT_TEST_REPORT']).write_text(json.dumps(report,indent=2))
        print('EQUIVALENCE_PASSED',json.dumps(report),flush=True)

if __name__=='__main__':unittest.main(verbosity=2)

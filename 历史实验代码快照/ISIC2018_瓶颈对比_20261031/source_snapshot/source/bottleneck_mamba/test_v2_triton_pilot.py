"""GPU contract for direct V2 Triton scan; includes tiny/zero-delta regression."""
from pathlib import Path
import sys
import unittest
import json
import os
P=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(P),str(P/'source/comparison')]
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t')
import torch
from Models.TinyViM.mamba_bottleneck_v2 import scan_without_delta_injection,BottleneckMambaV2

class DirectScanContract(unittest.TestCase):
    def test_forward_and_gradients_match_without_division(self):
        self.assertTrue((P/'Models/TinyViM/mamba_bottleneck_v2_triton_pilot.py').exists(),'Direct Triton V2 scan missing')
        from Models.TinyViM.mamba_bottleneck_v2_triton_pilot import scan_v2_triton,BottleneckMambaV2TritonPilot
        torch.set_num_threads(2)
        self.assertTrue(torch.cuda.is_available())
        records=[]
        for length,small in [(7,False),(64,False),(256,False),(7,True)]:
            torch.manual_seed(44)
            u=torch.randn(2,32,length,device='cuda')
            delta=torch.exp(torch.empty_like(u).uniform_(-9,-2)) if not small else torch.full_like(u,1e-25)
            if small:delta[:,:,0]=0
            vals=[u,delta,-torch.rand(32,16,device='cuda')*4-0.1,torch.randn(2,16,length,device='cuda'),torch.randn(2,16,length,device='cuda')]
            left=[x.detach().clone().requires_grad_() for x in vals];right=[x.detach().clone().requires_grad_() for x in vals]
            a=scan_without_delta_injection(*left);b=scan_v2_triton(*right)
            torch.testing.assert_close(a,b,rtol=3e-4,atol=5e-4)
            weight=torch.randn_like(a);(a*weight).mean().backward();(b*weight).mean().backward()
            errors={}
            for n,x,y in zip(['u','delta','A','B','C'],left,right):
                self.assertTrue(torch.isfinite(y.grad).all())
                torch.testing.assert_close(x.grad,y.grad,rtol=3e-3,atol=5e-4)
                errors[n]=float((x.grad-y.grad).abs().max())
            records.append({'length':length,'tiny_and_zero_delta':small,'max_output_abs':float((a-b).abs().max()),'gradient_max_abs':errors})
        old=BottleneckMambaV2(512,16).cuda();new=BottleneckMambaV2TritonPilot(512,16).cuda()
        new.load_state_dict(old.state_dict(),strict=True)
        x=torch.randn(1,512,4,4,device='cuda',requires_grad=True);y=x.detach().clone().requires_grad_()
        a=old(x);b=new(y);torch.testing.assert_close(a,b,rtol=5e-4,atol=5e-4)
        a.square().mean().backward();b.square().mean().backward()
        torch.testing.assert_close(x.grad,y.grad,rtol=3e-3,atol=5e-4)
        for (na,pa),(nb,pb) in zip(old.named_parameters(),new.named_parameters()):
            self.assertEqual(na,nb);self.assertIsNotNone(pb.grad)
            torch.testing.assert_close(pa.grad,pb.grad,rtol=3e-3,atol=5e-4)
        report={'status':'passed','checks':records,'module_all_parameter_gradients':True,'strict_state_dict':True,'parameters':sum(p.numel() for p in new.parameters())}
        if os.environ.get('CUDA_PILOT_TEST_REPORT'):Path(os.environ['CUDA_PILOT_TEST_REPORT']).write_text(json.dumps(report,indent=2))
        print('DIRECT_TRITON_EQUIVALENCE_PASSED',json.dumps(report),flush=True)

if __name__=='__main__':unittest.main(verbosity=2)

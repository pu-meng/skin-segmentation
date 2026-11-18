import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['nnUNet_compile']='false'
import torch
from tta import SubsetPredictor, VIEWS
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.calls=0
    def forward(self,x):
        self.calls+=1
        # Position dependent: deliberately not flip equivariant.
        z=torch.arange(x.shape[-3],dtype=x.dtype).reshape(1,1,-1,1,1)
        return torch.cat((x+z,x.square()-z,2*x+3*z),1)
def main():
    torch.set_num_threads(1); torch.manual_seed(17)
    x=torch.randn(1,1,5,6,7)
    results=[]
    for n in (1,2,4,8):
        p=SubsetPredictor(views=n,device=torch.device('cpu')); p.network=Toy()
        p.allowed_mirroring_axes=(0,1,2)
        y=p._internal_maybe_mirror_and_predict(x)
        assert p.network.calls==n, (n,p.network.calls)
        net=Toy(); expected=None
        for axes in VIEWS[n]:
            dims=tuple(a+2 for a in axes)
            value=net(torch.flip(x,dims) if dims else x)
            value=torch.flip(value,dims) if dims else value
            expected=value.clone() if expected is None else expected+value
        expected/=n
        torch.testing.assert_close(y,expected,rtol=0,atol=0)
        assert y.shape==(1,3,5,6,7)
        results.append({'views':n,'calls':p.network.calls,'alignment_max_abs':float((y-expected).abs().max())})
    p=SubsetPredictor(views=8,device=torch.device('cpu')); p.network=Toy(); p.allowed_mirroring_axes=(0,1,2)
    original=nnUNetPredictor(device=torch.device('cpu')); original.network=Toy(); original.allowed_mirroring_axes=(0,1,2)
    a=p._internal_maybe_mirror_and_predict(x); b=original._internal_maybe_mirror_and_predict(x)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert not torch.cuda.is_initialized()
    import json
    print(json.dumps({'status':'passed','views':results,'original_eight_max_abs':float((a-b).abs().max()),'cuda_initialized':torch.cuda.is_initialized()}))
if __name__=='__main__': main()

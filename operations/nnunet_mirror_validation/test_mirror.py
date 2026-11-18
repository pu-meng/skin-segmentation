import torch
from pathlib import Path
import ast,itertools
try:
 from mirror_core import mirror_logits
except ImportError:
 mirror_logits=None

def test_mirror():
 assert mirror_logits is not None, 'mirror_logits not implemented'
 x=torch.arange(30,dtype=torch.float32).reshape(1,1,5,6)/10
 calls=[]
 def net(z):
  calls.append(z.clone())
  return z.square()+torch.arange(6).reshape(1,1,1,6)*z+torch.arange(5).reshape(1,1,5,1)
 expected=sum(torch.flip(net(torch.flip(x,d)),d) if d else net(x) for d in [(),(2,),(3,),(2,3)])/4
 calls.clear(); got=mirror_logits(net,x,True)
 torch.testing.assert_close(got,expected,rtol=0,atol=0)
 assert len(calls)==4
 for a,d in zip(calls,[(),(2,),(3,),(2,3)]):
  assert torch.equal(a,torch.flip(x,d) if d else x)
 calls.clear();torch.testing.assert_close(mirror_logits(net,x,False),net(x),rtol=0,atol=0)
 assert len(calls)==2
 # Independently execute installed nnU-Net's exact method on the toy.
 s=Path('/home/PuMengYu/nnUNet/nnunetv2/inference/predict_from_raw_data.py').read_text()
 n=next(n for n in ast.walk(ast.parse(s)) if isinstance(n,ast.FunctionDef) and n.name=='_internal_maybe_mirror_and_predict')
 ns={'torch':torch,'itertools':itertools};exec(compile(ast.Module(body=[n],type_ignores=[]),'installed_nnunet','exec'),ns)
 class P: pass
 p=P();p.allowed_mirroring_axes=(0,1);p.use_mirroring=True;p.network=net
 torch.testing.assert_close(got,ns[n.name](p,x),rtol=0,atol=0)
 assert not torch.cuda.is_initialized()
 print('PASS non-equivariant toy: exact four views, inverse axes, logit fusion, none identity, installed nnUNet compatibility; CPU only')
if __name__=='__main__':test_mirror()

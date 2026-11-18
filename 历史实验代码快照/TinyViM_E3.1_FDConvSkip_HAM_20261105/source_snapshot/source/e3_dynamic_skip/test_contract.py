import sys,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
def test_missing():
 try:
  from Models.E3DynamicSkip.model import DynamicSkip
 except ImportError:
  raise AssertionError('E3.1 DynamicSkip implementation missing')
 m=DynamicSkip().eval();s=torch.randn(2,64,8,8);d=torch.randn_like(s)
 assert m(s,d).shape==s.shape
 assert torch.equal(m(s,d,enabled=False),s)
if __name__=='__main__':test_missing()

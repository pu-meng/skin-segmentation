from pathlib import Path
import ast,hashlib
R=Path('/home/PuMengYu/experiments/skin_segmentation');D=R/'source/three_module_impl_20261104';E=Path('/home/PuMengYu/论文汇总/TinyViM三模块/evidence_20261104_200531')
s=(E/'official_classification_models_waveformer.py').read_text();t=ast.parse(s)
names={'_fft2d','_ifft2d','_fourier_lambda_map','_inv_softplus','WaveOperator'}
parts=[ast.get_source_segment(s,n) for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
(D/'official_wave.py').write_text('import math\nimport torch\nfrom torch import nn\nfrom torch.nn import functional as F\nfrom timm.models.layers import trunc_normal_\n\n'+'\n\n'.join(parts)+'\n')
c=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='WaveOperator');f=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='forward');f.decorator_list=[]
fwd=ast.unparse(f).replace('ln_lam.view(1, 1, H, Wr)','ln_lam.view(K, D, H, Wr)')
# Inherited forward differs in only the channel-aware lambda view. FFT propagation is FP32.
(D/'candidate.py').write_text('''"""Isolated E1. HSA/Skip remain blocked, not placeholder implementations."""
import math
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F
from official_wave import WaveOperator, _fft2d
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]

class DirectionalWaveOperator(WaveOperator):
 def __init__(self,**kw):
  super().__init__(**kw)
  # Implementation choices: fixed time 1 (official default), K1 (official base).
  # t parameters replaced by constant buffers: no trainable propagation time.
  del self.t_raw;del self.t_ch
  self.register_buffer('t_raw',torch.full((self.num_heads,),math.log(math.expm1(1.))))
  self.register_buffer('t_ch',torch.zeros(self.hidden_dim))
  self.direction_raw=nn.Parameter(torch.zeros(self.hidden_dim))
 def direction_coefficients(self):
  # Positive bounded parameterization avoids a negative fractional Laplacian.
  a=torch.tanh(self.direction_raw)
  return 1+a,1-a
 def _get_lambda(self,H,W,device):
  ax,ay=self.direction_coefficients()
  kx=2*math.pi*torch.fft.rfftfreq(W,device=device)
  ky=2*math.pi*torch.fft.fftfreq(H,device=device)
  return (ax[:,None,None]*kx[None,None,:].square()+ay[:,None,None]*ky[None,:,None].square()).reshape(self.num_heads,self.head_dim,H,W//2+1)
 def forward(self,x,alpha_raw=None):
  # Locally disable autocast: all FFTs, exponents, norm and complex work FP32.
  with torch.autocast(device_type=x.device.type,enabled=False):
   if alpha_raw is not None:
    H,W=x.shape[-2:];idx=(-torch.arange(H,device=x.device))%H
    alpha_raw=alpha_raw.float()
    # Real rFFT boundary constraint only; nonboundary spectral field unchanged.
    mask=torch.zeros((1,W//2+1,1),device=x.device);mask[:,0]=1
    if W%2==0:mask[:,-1]=1
    alpha_raw=alpha_raw*(1-mask)+(alpha_raw+alpha_raw[idx])*0.5*mask
   return self.propagate(x.float(),alpha_raw)
'''+ '\n'.join(' '+line for line in fwd.replace('def forward(','def propagate(').splitlines())+'''

class DStage(nn.Module):
 def __init__(self):
  super().__init__()
  self.wave=DirectionalWaveOperator(res=8,dim=512,hidden_dim=512,num_heads=1)
  # Source-owned alpha_embed moved into the single independent decoder stage.
  self.alpha_embed=nn.Parameter(torch.full((8,5,1),math.log(math.expm1(.05))))
  self.gamma=nn.Parameter(torch.full((1,512,1,1),.001))
 def forward(self,x):
  # Official DW3 and output Linear already supply entry/output convolutions.
  # No duplicated PW; replace complete DoubleConv D3, preserve all other stages.
  return x+self.gamma*self.wave(x,self.alpha_embed)

def build_candidate(pretrained=True,d=False,h=False,s=False):
 if h or s:raise NotImplementedError('HSA/Skip core gate and SS2D definition unresolved; no surrogate')
 from Models.TinyViM.four_stage_decoder import build_four_stage_baseline
 model=build_four_stage_baseline(pretrained=pretrained)
 if d:
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(42);model.decoder.stages[3]=DStage()
 return model
''')
print('WROTE',D/'candidate.py')

"""Isolated E1. HSA/Skip remain blocked, not placeholder implementations."""
import math
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F
from source.wpo_mr_20261105.official_wave import WaveOperator, _fft2d
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
 def propagate(self, x: torch.Tensor, alpha_raw=None):
     if alpha_raw is None:
         raise ValueError('WaveOperator.forward requires alpha_raw (stage-shared parameter of shape (res, res//2+1, K)). It is supplied by WaveFormer.forward_features via AdditionalInputSequential.')
     H, W = x.shape[-2:]
     B, C = (x.shape[0], self.hidden_dim)
     K, D = (self.num_heads, self.head_dim)
     u0, v0, z = self._project(x)
     x_u0 = _fft2d(u0)
     x_v0 = _fft2d(v0)
     Wr = x_u0.shape[-1]
     lam = self._get_lambda(H, W, u0.device)
     ln_lam = lam.clamp(min=1e-12).log()
     p = self.spectral_exponent.view(K, D, 1, 1)
     lam_p = torch.exp(p * ln_lam.view(K, D, H, Wr))
     t = self.propagation_time.view(K, D, 1, 1)
     alpha_ch_kd = self.alpha_ch.view(K, D)
     alpha = F.softplus(alpha_raw.unsqueeze(-1) + alpha_ch_kd.view(1, 1, K, D)).clamp(min=WaveOperator.ALPHA_MIN)
     alpha = alpha.permute(2, 3, 0, 1)
     resp_damp = torch.exp(-alpha * lam_p * t)
     omega = torch.sqrt(lam_p.clamp(min=1e-12))
     omega_safe = omega.clamp(min=1e-06)
     phase = omega * t
     envelope = torch.exp(-alpha * lam_p * t / 2.0)
     resp_cos = envelope * torch.cos(phase)
     resp_sin = envelope * torch.sin(phase) / omega_safe
     spec_dtype = x_u0.dtype
     resp_damp = resp_damp.to(spec_dtype)
     resp_cos = resp_cos.to(spec_dtype)
     resp_sin = resp_sin.to(spec_dtype)
     x_u0 = x_u0.view(B, K, D, H, Wr)
     x_v0 = x_v0.view(B, K, D, H, Wr)
     u_t_damp = resp_damp * x_u0
     u_t_wave = resp_cos * x_u0 + resp_sin * (x_v0 + alpha * lam_p / 2.0 * x_u0)
     w = self.wave_weight.view(K, 1, 1, 1)
     u_t = u_t_damp + w * (u_t_wave - u_t_damp)
     u_t = u_t.reshape(B, C, H, Wr)
     return self._gate_and_project_out(u_t, z, (H, W))

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

"""Independent engineering pilot: preserve V2 equations using existing CUDA scan.

The standard kernel injects delta*B*u. Supply u=T/delta, D=None, z=None,
so its injection equals B*T; apply D*T and SiLU(z) outside using original T.
No new parameters, no package edits, no modification of active V2 source.
Numerical equality must be tested: division/cancellation is not bitwise exact.
"""
import torch
from torch.nn import functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn,selective_scan_ref
from Models.TinyViM.mamba_bottleneck_v2 import NoDeltaInjectionMamba,BottleneckMambaV2


def scan_v2_cuda_equivalent(u,delta,A,B,C):
    original_dtype=u.dtype
    uf,df=u.float(),delta.float()
    scan=selective_scan_fn if u.is_cuda else selective_scan_ref
    y=scan((uf/df).contiguous(),df.contiguous(),A.float().contiguous(),
           B.float().contiguous(),C.float().contiguous(),D=None,z=None,
           delta_bias=None,delta_softplus=False)
    return y.to(original_dtype)


class CudaEquivalentMamba(NoDeltaInjectionMamba):
    def forward(self,hidden_states,inference_params=None):
        if inference_params is not None:
            raise NotImplementedError('Pilot supports full image sequences only')
        x,z=self.in_proj(hidden_states).transpose(1,2).chunk(2,dim=1)
        u=self.act(self.conv1d(x)[...,:hidden_states.shape[1]])
        q=self.x_proj(u.transpose(1,2))
        rho,B,C=torch.split(q,[self.dt_rank,self.d_state,self.d_state],dim=-1)
        delta=F.softplus(self.dt_proj(rho)).transpose(1,2)
        A=-torch.exp(self.A_log.float())
        y=scan_v2_cuda_equivalent(u,delta,A,B.transpose(1,2),C.transpose(1,2))
        y=(y+self.D[None,:,None]*u)*F.silu(z)
        return self.out_proj(y.transpose(1,2))


class BottleneckMambaV2CudaPilot(BottleneckMambaV2):
    def __init__(self,channels=512,d_state=16,seed=44):
        super().__init__(channels,d_state,seed)
        # Same constructor and parameter names; replace class behavior only.
        self.mixer.__class__=CudaEquivalentMamba

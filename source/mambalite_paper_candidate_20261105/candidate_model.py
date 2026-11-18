"""Figure 2(b) reconstruction, not a strict author reproduction.
Frozen public surrounding network; SS2D ratio=1, DWConv3 then SiLU,
independent directional scan parameters, sum, LN, gated output projection.
Only AMF skip_scale starts at zero; its residual operand is raw X_k.
"""
import os, sys, math, hashlib
from pathlib import Path
import torch
from torch import nn
from einops import repeat
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'source/comparison'))
from ham_fast_runner import baseline_api
api=baseline_api()
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

def scan_dispatch(u,delta,A,B,C,D,delta_bias,delta_softplus,nrows,cpu_reference=False):
    if u.device.type=='cpu':
        if not cpu_reference or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
            raise RuntimeError('CPU scan prohibited outside explicit hidden-CUDA CPU acceptance')
        fn=selective_scan_ref
    else:
        if cpu_reference: raise RuntimeError('CPU reference flag prohibited for CUDA execution')
        fn=selective_scan_fn
    return fn(u,delta,A,B,C,D=D,z=None,delta_bias=delta_bias,delta_softplus=delta_softplus)

ss=(HERE/'upstream_SS2D_class_snapshot.py').read_text()
ss=ss.replace('return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)',
              'return scan_dispatch(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, cpu_reference=self.cpu_reference)')
ns=dict(torch=torch,nn=nn,math=math,repeat=repeat,scan_dispatch=scan_dispatch)
exec(compile(ss,str(HERE/'upstream_SS2D_class_snapshot.py'),'exec'),ns)
SS2D=ns['SS2D']

class Figure2bSSMLayer(nn.Module):
    def __init__(self,d_model,d_state,d_conv,expand):
        super().__init__()
        self.mamba=SS2D(d_model=d_model,d_state=d_state,ssm_ratio=1.,ssm_rank_ratio=1.,d_conv=3,forward_type='v0')
        self.mamba.cpu_reference=False
    def forward(self,x):
        if x.ndim==4:
            b,c,h,w=x.shape
            grid=x.permute(0,2,3,1).contiguous()
        else:
            b,n,c=x.shape
            h=w=math.isqrt(n)
            if h*w!=n:raise ValueError('AMF token interface requires square spatial shape')
            grid=x.reshape(b,h,w,c)
        return self.mamba(grid).reshape(b,h*w,c)

# Keep a literal source snapshot; never patch/import-modify the shared model.
source=(HERE/'public_network_snapshot.py').read_text()
start=source.index('class MambaSSMLayer(')
end=source.index('# Local-Global Feature Mixing Module',start)
source=source[:start]+'MambaSSMLayer = Figure2bSSMLayer\n\n'+source[end:]
source=source.replace('self.skip_scale = nn.Parameter(torch.ones(1))','self.skip_scale = nn.Parameter(torch.zeros(1))')
source=source.replace('x_chunks = torch.chunk(x_norm, self.num_branches, dim=2)',
    'x_chunks = torch.chunk(x_norm, self.num_branches, dim=2)\n        raw_chunks = torch.chunk(x, self.num_branches, dim=2)')
source=source.replace('self.skip_scale * x_chunk','self.skip_scale * raw_chunks[i]')
# Attention weights are discarded in the public code. SDPA avoids materializing
# the huge attention matrix; same attention operation, FP32, no AMP/checkpointing.
source=source.replace('self.global_attn(x, x, x)','self.global_attn(x, x, x, need_weights=False)')
space={'Figure2bSSMLayer':Figure2bSSMLayer,'__name__':'figure2b_frozen_surroundings'}
exec(compile(source,str(HERE/'public_network_snapshot.py'),'exec'),space)

def build_model(cpu_reference=False):
    if cpu_reference and (os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized()):
        raise RuntimeError('CPU acceptance must start with CUDA hidden')
    m=space['MambaLiteUNet'](c_list=[16,32,48,64,96,128])
    for op in m.modules():
        if isinstance(op,SS2D):op.cpu_reference=cpu_reference
    return m

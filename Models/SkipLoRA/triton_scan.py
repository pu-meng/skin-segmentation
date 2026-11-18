"""Independent V2 scan with adjacent-lane forward reduction and libdevice exp.

FP32 full-sequence forward/backward. One GPU program per batch/channel holds
16 states; backward uses atomics to reduce shared B/C and A gradients.
Not a work-efficient parallel-prefix implementation; removes Python-token
launches by keeping the recurrence inside compiled GPU programs.
"""
import torch
import triton
import triton.language as tl
from torch.nn import functional as F
from Models.TinyViM.mamba_bottleneck_v2 import NoDeltaInjectionMamba,BottleneckMambaV2,scan_without_delta_injection


@triton.jit
def _forward(U,D,A,B,C,H,Y,E:tl.constexpr,N:tl.constexpr,L:tl.constexpr,S:tl.constexpr):
    p=tl.program_id(0);batch=p//E;channel=p%E
    n=tl.arange(0,S);mask=n<N
    a=tl.load(A+channel*N+n,mask,other=0)
    state=tl.full((S,),0,tl.float32)
    for t in range(0,L):
        u=tl.load(U+p*L+t);delta=tl.load(D+p*L+t)
        b=tl.load(B+batch*N*L+n*L+t,mask,other=0)
        c=tl.load(C+batch*N*L+n*L+t,mask,other=0)
        keep=tl.math.exp(delta*a)
        state=keep*state+b*u
        tl.store(H+p*L*N+t*N+n,state,mask)
        product=state*c
        # ATen contiguous N=16 sum combines adjacent lanes first.
        if N == 16:
            v8=tl.sum(tl.reshape(product,(8,2)),axis=1)
            v4=tl.sum(tl.reshape(v8,(4,2)),axis=1)
            v2=tl.sum(tl.reshape(v4,(2,2)),axis=1)
            out=tl.sum(v2,axis=0)
        else:
            out=tl.sum(product,axis=0)
        tl.store(Y+p*L+t,out)


@triton.jit
def _backward(U,D,A,B,C,H,DY,DU,DD,DA,DB,DC,E:tl.constexpr,N:tl.constexpr,L:tl.constexpr,S:tl.constexpr):
    p=tl.program_id(0);batch=p//E;channel=p%E
    n=tl.arange(0,S);mask=n<N
    a=tl.load(A+channel*N+n,mask,other=0)
    adj=tl.full((S,),0,tl.float32)
    grad_a=tl.full((S,),0,tl.float32)
    for reverse in range(0,L):
        t=L-1-reverse
        u=tl.load(U+p*L+t);delta=tl.load(D+p*L+t)
        b=tl.load(B+batch*N*L+n*L+t,mask,other=0)
        c=tl.load(C+batch*N*L+n*L+t,mask,other=0)
        state=tl.load(H+p*L*N+t*N+n,mask,other=0)
        previous=tl.load(H+p*L*N+(t-1)*N+n,mask & (t>0),other=0)
        dy=tl.load(DY+p*L+t)
        keep=tl.math.exp(delta*a)
        adj=adj+dy*c
        tl.store(DU+p*L+t,tl.sum(adj*b,axis=0))
        transition_grad=adj*previous*keep
        tl.store(DD+p*L+t,tl.sum(transition_grad*a,axis=0))
        grad_a=grad_a+transition_grad*delta
        tl.atomic_add(DB+batch*N*L+n*L+t,adj*u,mask,sem='relaxed')
        tl.atomic_add(DC+batch*N*L+n*L+t,dy*state,mask,sem='relaxed')
        adj=adj*keep
    tl.atomic_add(DA+channel*N+n,grad_a,mask,sem='relaxed')


class _DirectScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx,u,delta,A,B,C):
        batch,E,L=u.shape;N=A.shape[1]
        states=torch.empty((batch,E,L,N),device=u.device,dtype=torch.float32)
        y=torch.empty_like(u)
        _forward[(batch*E,)](u,delta,A,B,C,states,y,E,N,L,triton.next_power_of_2(N),num_warps=1,enable_fp_fusion=False)
        ctx.save_for_backward(u,delta,A,B,C,states)
        return y

    @staticmethod
    def backward(ctx,dy):
        u,delta,A,B,C,states=ctx.saved_tensors
        batch,E,L=u.shape;N=A.shape[1]
        du=torch.empty_like(u);dd=torch.empty_like(delta)
        da=torch.zeros_like(A);db=torch.zeros_like(B);dc=torch.zeros_like(C)
        _backward[(batch*E,)](u,delta,A,B,C,states,dy.contiguous(),du,dd,da,db,dc,E,N,L,
                             triton.next_power_of_2(N),num_warps=1,enable_fp_fusion=False)
        return du,dd,da,db,dc


def scan_v2_triton(u,delta,A,B,C):
    if not u.is_cuda:return scan_without_delta_injection(u,delta,A,B,C)
    values=[v.float().contiguous() for v in (u,delta,A,B,C)]
    return _DirectScan.apply(*values).to(u.dtype)


class DirectTritonMamba(NoDeltaInjectionMamba):
    def forward(self,hidden_states,inference_params=None):
        if inference_params is not None:raise NotImplementedError('Full image sequence pilot only')
        x,z=self.in_proj(hidden_states).transpose(1,2).chunk(2,dim=1)
        u=self.act(self.conv1d(x)[...,:hidden_states.shape[1]])
        q=self.x_proj(u.transpose(1,2))
        rho,B,C=torch.split(q,[self.dt_rank,self.d_state,self.d_state],dim=-1)
        delta=F.softplus(self.dt_proj(rho)).transpose(1,2)
        A=-torch.exp(self.A_log.float())
        y=scan_v2_triton(u,delta,A,B.transpose(1,2),C.transpose(1,2))
        y=(y+self.D[None,:,None]*u)*F.silu(z)
        return self.out_proj(y.transpose(1,2))


class BottleneckMambaV2TritonPilot(BottleneckMambaV2):
    def __init__(self,channels=512,d_state=16,seed=44):
        super().__init__(channels,d_state,seed)
        self.mixer.__class__=DirectTritonMamba

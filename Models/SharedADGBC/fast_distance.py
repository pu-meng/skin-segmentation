"""FP32 bounded-memory diagonal distances with fused Triton backward, no atomics.

Strided image-token forward uses bounded ATen chunks to preserve its arithmetic;
other layouts use fused Triton forward. Input backward fuses differences/reductions.
Parameter backward
uses small [ceil(B*N/128),K,D] partials followed by a deterministic reduction.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _forward(Z,C,S,O,P:tl.constexpr,K:tl.constexpr,D:tl.constexpr,
             BP:tl.constexpr,BK:tl.constexpr,BD:tl.constexpr):
    p=tl.program_id(0)*BP+tl.arange(0,BP)
    k=tl.arange(0,BK);d=tl.arange(0,BD)
    z=tl.load(Z+p[:,None,None]*D+d[None,None,:],(p[:,None,None]<P)&(d[None,None,:]<D),0)
    c=tl.load(C+k[None,:,None]*D+d[None,None,:],(k[None,:,None]<K)&(d[None,None,:]<D),0)
    s=tl.load(S+k[None,:,None]*D+d[None,None,:],(k[None,:,None]<K)&(d[None,None,:]<D),1)
    v=tl.fdiv(z-c,s,ieee_rounding=True)
    out=tl.sum(v*v,axis=2)
    tl.store(O+p[:,None]*K+k[None,:],out,(p[:,None]<P)&(k[None,:]<K))


@triton.jit
def _input_backward(Z,C,S,G,DZ,P:tl.constexpr,K:tl.constexpr,D:tl.constexpr,
                    BP:tl.constexpr,BK:tl.constexpr,BD:tl.constexpr):
    p=tl.program_id(0)*BP+tl.arange(0,BP)
    k=tl.arange(0,BK);d=tl.arange(0,BD)
    z=tl.load(Z+p[:,None,None]*D+d[None,None,:],(p[:,None,None]<P)&(d[None,None,:]<D),0)
    c=tl.load(C+k[None,:,None]*D+d[None,None,:],(k[None,:,None]<K)&(d[None,None,:]<D),0)
    s=tl.load(S+k[None,:,None]*D+d[None,None,:],(k[None,:,None]<K)&(d[None,None,:]<D),1)
    g=tl.load(G+p[:,None]*K+k[None,:],(p[:,None]<P)&(k[None,:]<K),0)
    value=2.*g[:,:,None]*((z-c)/s)/s
    dz=tl.sum(value,axis=1)
    tl.store(DZ+p[:,None]*D+d[None,:],dz,(p[:,None]<P)&(d[None,:]<D))


@triton.jit
def _parameter_partials(Z,C,S,G,PC,PS,P:tl.constexpr,K:tl.constexpr,D:tl.constexpr,
                        BT:tl.constexpr,BD:tl.constexpr):
    tile=tl.program_id(0);k=tl.program_id(1)
    p=tile*BT+tl.arange(0,BT);d=tl.arange(0,BD)
    z=tl.load(Z+p[:,None]*D+d[None,:],(p[:,None]<P)&(d[None,:]<D),0)
    c=tl.load(C+k*D+d,d<D,0)
    s=tl.load(S+k*D+d,d<D,1)
    g=tl.load(G+p*K+k,p<P,0)
    normalized=(z-c[None,:])/s[None,:]
    dc=tl.sum(-2.*g[:,None]*normalized/s[None,:],axis=0)
    ds=tl.sum(-2.*g[:,None]*normalized*normalized/s[None,:],axis=0)
    tl.store(PC+(tile*K+k)*D+d,dc,d<D)
    tl.store(PS+(tile*K+k)*D+d,ds,d<D)


@triton.jit
def _reduce_parameters(PC,PS,DC,DS,K:tl.constexpr,D:tl.constexpr,T:tl.constexpr,
                       TB:tl.constexpr,BD:tl.constexpr):
    k=tl.program_id(0);t=tl.arange(0,TB);d=tl.arange(0,BD)
    mask=(t[:,None]<T)&(d[None,:]<D)
    pc=tl.load(PC+(t[:,None]*K+k)*D+d[None,:],mask,0)
    ps=tl.load(PS+(t[:,None]*K+k)*D+d[None,:],mask,0)
    tl.store(DC+k*D+d,tl.sum(pc,axis=0),d<D)
    tl.store(DS+k*D+d,tl.sum(ps,axis=0),d<D)


class _Distance(torch.autograd.Function):
    @staticmethod
    def forward(ctx,z,c,s,layout_preserving):
        B,N,D=z.shape;K=c.shape[0];P=B*N
        out=torch.empty((B,N,K),device=z.device,dtype=torch.float32)
        if layout_preserving:
            # Keep ATen's arithmetic and strided reduction order at initialization,
            # where almost-equal ball logits are particularly sensitive to ulps.
            # Autograd is disabled inside this custom forward. Only one bounded
            # chunk is materialized and no [B,N,K,D] tensor is retained for backward.
            for start in range(0,N,512):
                part=z[:,start:start+512]
                out[:,start:start+512]=(((part[:,:,None]-c[None,None])/s[None,None])**2).sum(-1)
        else:
            _forward[(triton.cdiv(P,4),)](z,c,s,out,P,K,D,4,triton.next_power_of_2(K),
                                        triton.next_power_of_2(D),num_warps=4,enable_fp_fusion=False)
        ctx.save_for_backward(z.contiguous(),c,s)
        return out

    @staticmethod
    def backward(ctx,grad):
        z,c,s=ctx.saved_tensors;B,N,D=z.shape;P=B*N;K=c.shape[0]
        grad=grad.contiguous();dz=torch.empty_like(z);dc=torch.empty_like(c);ds=torch.empty_like(s)
        tiles=triton.cdiv(P,128)
        pc=torch.empty((tiles,K,D),device=z.device,dtype=torch.float32)
        ps=torch.empty_like(pc)
        _input_backward[(triton.cdiv(P,4),)](z,c,s,grad,dz,P,K,D,4,triton.next_power_of_2(K),
            triton.next_power_of_2(D),num_warps=4,enable_fp_fusion=False)
        _parameter_partials[(tiles,K)](z,c,s,grad,pc,ps,P,K,D,128,triton.next_power_of_2(D),
            num_warps=4,enable_fp_fusion=False)
        _reduce_parameters[(K,)](pc,ps,dc,ds,K,D,tiles,triton.next_power_of_2(tiles),
            triton.next_power_of_2(D),num_warps=8,enable_fp_fusion=False)
        return dz,dc,ds,None


def direct_distance(z,centers,sigma):
    if z.dtype!=torch.float32 or centers.dtype!=torch.float32 or sigma.dtype!=torch.float32:
        raise TypeError('Independent fast backend currently supports FP32 only')
    if z.ndim!=3 or centers.ndim!=2 or sigma.shape!=centers.shape or z.shape[-1]!=centers.shape[-1]:
        raise ValueError('Expected z[B,N,D], centers/sigma[K,D]')
    if not z.is_cuda:
        return (((z.unsqueeze(2)-centers[None,None])/sigma[None,None])**2).sum(-1)
    preserve=z.stride(1)==1 and z.shape[-1]==64 and z.shape[1]>=512
    return _Distance.apply(z if preserve else z.contiguous(),centers.contiguous(),sigma.contiguous(),preserve)


def quadratic_distance(z,centers,sigma):
    """Algebraic matmul candidate for numerical comparison; not the default backend."""
    inv=sigma.square().reciprocal()
    return z.square()@inv.T-2.*(z@(centers*inv).T)+(centers.square()*inv).sum(-1)

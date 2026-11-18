"""A1-S1 + Skip1-v0.6: replace only the shallow encoder-decoder skip block."""
import torch
from torch import nn
from torch.nn import functional as F
from source.a1_s1_20261108.candidate import build_candidate as build_a1_s1


def conv1(ci, co, bias=True):
    return nn.Conv2d(ci, co, 1, bias=bias)


class LocalRepDSC(nn.Module):
    """MobileOne-style depthwise reparameterizable local block (training form)."""
    def __init__(self, c=64):
        super().__init__()
        self.dw3 = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        self.bn3 = nn.BatchNorm2d(c)
        self.dw1 = nn.Conv2d(c, c, 1, bias=False, groups=c)
        self.bn1 = nn.BatchNorm2d(c)
        self.identity_bn = nn.BatchNorm2d(c)
        self.act1 = nn.GELU()
        self.pw = nn.Conv2d(c, c, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(c)
        self.act2 = nn.GELU()
        self.deploy_dw = None

    @staticmethod
    def _fuse_branch(conv, bn):
        w=conv.weight.detach()
        if conv.bias is None: b=torch.zeros_like(bn.running_mean)
        else: b=conv.bias.detach()
        scale=bn.weight.detach()*torch.rsqrt(bn.running_var.detach()+bn.eps)
        return w*scale[:,None,None,None], (b-bn.running_mean.detach())*scale+bn.bias.detach()

    @torch.no_grad()
    def switch_to_deploy(self):
        if self.deploy_dw is not None:return
        w3,b3=self._fuse_branch(self.dw3,self.bn3)
        w1,b1=self._fuse_branch(self.dw1,self.bn1)
        w1=F.pad(w1,(1,1,1,1))
        c=self.identity_bn.num_features
        wi=torch.zeros_like(w3);wi[:,0,1,1]=1
        scale=self.identity_bn.weight.detach()*torch.rsqrt(self.identity_bn.running_var.detach()+self.identity_bn.eps)
        wi=wi*scale[:,None,None,None]
        bi=(-self.identity_bn.running_mean.detach())*scale+self.identity_bn.bias.detach()
        fused=nn.Conv2d(c,c,3,padding=1,groups=c,bias=True,device=w3.device,dtype=w3.dtype)
        fused.weight.copy_(w3+w1+wi);fused.bias.copy_(b3+b1+bi)
        fused.requires_grad_(False)
        self.deploy_dw=fused
        self.dw3=self.bn3=self.dw1=self.bn1=self.identity_bn=None

    def forward(self, x):
        if self.deploy_dw is None:
            local = self.bn3(self.dw3(x)) + self.bn1(self.dw1(x)) + self.identity_bn(x)
        else:
            local = self.deploy_dw(x)
        return self.act2(self.pw_bn(self.pw(self.act1(local))))


class HaarLocalBlock(nn.Module):
    """Channel squeeze -> fixed Haar bands -> per-band DSC -> inverse Haar."""
    def __init__(self, c=64, hidden=16):
        super().__init__()
        self.reduce = conv1(c, hidden)
        self.band_ops = nn.ModuleList([
            nn.Sequential(nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
                          nn.GroupNorm(4, hidden), nn.Conv2d(hidden, hidden, 1), nn.GELU())
            for _ in range(4)
        ])
        self.expand = conv1(hidden, c)
        self.local = LocalRepDSC(c)
        self.fuse = conv1(c, c)

    @staticmethod
    def dwt(x):
        a=x[...,0::2,0::2]; b=x[...,0::2,1::2]
        c=x[...,1::2,0::2]; d=x[...,1::2,1::2]
        return ((a+b+c+d)*.5, (a-b+c-d)*.5, (a+b-c-d)*.5, (a-b-c+d)*.5)

    @staticmethod
    def idwt(bands):
        ll,lh,hl,hh=bands
        a=(ll+lh+hl+hh)*.5; b=(ll-lh+hl-hh)*.5
        c=(ll+lh-hl-hh)*.5; d=(ll-lh-hl+hh)*.5
        # Arrange each 2x2 Haar cell as [a b; c d], then interleave both
        # spatial axes. Stacking at -3 alone leaves an extra rank-5 axis.
        rows=torch.stack((torch.stack((a,b),-1),torch.stack((c,d),-1)),-3)
        n,ch,h,_,w,_=rows.shape
        return rows.reshape(n,ch,h*2,w*2)

    def forward(self, s):
        z=self.reduce(s)
        bands=self.dwt(z)
        z=self.idwt(tuple(op(v) for op,v in zip(self.band_ops,bands)))
        f_haar=self.expand(z)
        l_haar=self.local(s)
        return self.fuse(f_haar+l_haar)


class DCTLocalBlock(nn.Module):
    """Decoder-conditioned channel/frequency mask over an orthonormal 2-D DCT."""
    def __init__(self, c=64, size=64):
        super().__init__()
        n=torch.arange(size,dtype=torch.float32)
        k=n[:,None]
        mat=torch.cos(torch.pi*(2*n[None,:]+1)*k/(2*size))
        mat[0]*=(1/size)**.5
        mat[1:]*=(2/size)**.5
        self.register_buffer('dct_matrix',mat,persistent=True)
        self.channel_gate=nn.Sequential(nn.Linear(c,c),nn.GELU(),nn.Linear(c,c))
        self.freq_gate=nn.Conv2d(2*c,1,1)
        self.local=LocalRepDSC(c)
        self.fuse=conv1(c,c)

    def dct2(self, x):
        m=self.dct_matrix.to(device=x.device,dtype=x.dtype)
        return torch.matmul(m,torch.matmul(x,m.t()))

    def idct2(self, x):
        m=self.dct_matrix.to(device=x.device,dtype=x.dtype)
        return torch.matmul(m.t(),torch.matmul(x,m))

    def forward(self, s, d):
        z=self.dct2(s)
        # Per-frequency spatial mask from signed and magnitude coefficients.
        a_f=self.freq_gate(torch.cat((z,z.abs()),dim=1))
        a_c=self.channel_gate(d.mean((2,3))).unsqueeze(-1).unsqueeze(-1)
        mask=torch.tanh(a_c+a_f)
        f_dct=self.idct2(z*mask)
        l_dct=self.local(s)
        return self.fuse(f_dct+l_dct)


class CrossSourceInteraction(nn.Module):
    def __init__(self, c=64):
        super().__init__()
        self.haar=HaarLocalBlock(c)
        self.dct=DCTLocalBlock(c)
        self.content=conv1(2*c,c)
        self.spatial_dw=nn.Conv2d(3*c,3*c,3,padding=1,groups=3*c,bias=False)
        self.spatial_pw=nn.Conv2d(3*c,1,1)
        self.channel_gate=nn.Sequential(nn.Linear(c,16),nn.GELU(),nn.Linear(16,c))
        self.local_cross=LocalRepDSC(c)
        self.delta=conv1(c,c)
        # Small scalar LayerScale preserves the A1-S1 skip at initialization.
        self.gamma=nn.Parameter(torch.tensor(0.01))

    def forward(self, s, d):
        f_h=self.haar(s)
        f_c=self.dct(s,d)
        v=self.content(torch.cat((f_h,f_c),dim=1))
        a_s=self.spatial_pw(self.spatial_dw(torch.cat((s,d,s-d),dim=1)))
        a_c=self.channel_gate(d.mean((2,3))).unsqueeze(-1).unsqueeze(-1)
        g=torch.sigmoid(a_s+a_c)
        f_cross=v*g
        l_cross=self.local_cross(s)
        delta=self.delta(f_cross+l_cross)
        return s+self.gamma*delta


class Skip1V06(nn.Module):
    """Preserve A1-S1's D0 interface: return refined S and unchanged decoder D."""
    def __init__(self):
        super().__init__()
        self.interaction=CrossSourceInteraction(64)

    def forward(self, s, d):
        if s.shape != d.shape or s.ndim != 4 or s.shape[1:] != (64,64,64):
            raise ValueError(f'Skip1-v0.6 expects matching [B,64,64,64], got {tuple(s.shape)} and {tuple(d.shape)}')
        return self.interaction(s,d),d


def build_candidate(pretrained=True):
    model=build_a1_s1(pretrained=pretrained)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4606)
        model.decoder.e3=Skip1V06()
    return model

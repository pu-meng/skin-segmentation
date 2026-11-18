"""C0/C1 candidate-aware GLR and the real A1 D2 interface."""
import torch
from torch import nn
from torch.nn import functional as F

def gn(channels):
    groups = next((n for n in (8, 4, 2, 1) if channels % n == 0), None)
    if groups is None:
        raise ValueError('invalid GroupNorm channels')
    return nn.GroupNorm(groups, channels)

def dw(channels, kernel):
    return nn.Conv2d(channels, channels, kernel, padding=kernel//2, groups=channels, bias=False)

class GLR(nn.Module):
    def __init__(self, channels, variant):
        super().__init__()
        if variant not in ('C0','C1'):
            raise ValueError('Only C0/C1 are implemented')
        self.variant = variant
        self.pre = nn.Sequential(nn.Conv2d(channels,channels,1,bias=True),dw(channels,3),gn(channels),nn.GELU())
        self.branches = nn.ModuleList([nn.Sequential(dw(channels,3),gn(channels),nn.GELU()),
                                       nn.Sequential(dw(channels,7),gn(channels),nn.GELU()),
                                       nn.Sequential(nn.Conv2d(channels,channels,1,bias=True),gn(channels),nn.GELU())])
        d=max(8,channels//8)
        self.context_dw=dw(channels,3)
        self.context_project=nn.Conv2d(channels,d,1,bias=True)
        self.descriptors=nn.ModuleList([nn.Conv2d(channels,d,1,bias=True) for _ in range(3)])
        self.protect=nn.Sequential(dw(channels,3),gn(channels),nn.GELU(),nn.Conv2d(channels,channels,1,bias=True))
        self.refine=nn.Conv2d(channels,channels,1,bias=False)
        self.gamma_c=nn.Parameter(torch.tensor(.01))
        self.gamma_s=nn.Parameter(torch.tensor(.01))
        self.select_dw=dw(4*d,3)
        self.select_norm=gn(4*d)
        # Construct last so the three output rows shared by C0/C1 do not
        # perturb initialization of any other parameter.
        self.select_out=nn.Conv2d(4*d,4 if variant=='C1' else 3,1,bias=True)
        nn.init.zeros_(self.select_out.weight);nn.init.zeros_(self.select_out.bias)

    def forward_with_weights(self,x):
        if min(x.shape[-2:])<4:
            raise ValueError('GLR requires H,W >=4 for fixed context pooling')
        u=self.pre(x)
        branches=[branch(u) for branch in self.branches]
        context=self.context_project(self.context_dw(F.adaptive_avg_pool2d(u,(4,4))))
        context=F.interpolate(context,size=u.shape[-2:],mode='bilinear',align_corners=False)
        descriptors=[conv(v) for conv,v in zip(self.descriptors,branches)]
        weights=self.select_out(F.gelu(self.select_norm(self.select_dw(torch.cat([context,*descriptors],dim=1))))).softmax(dim=1)
        offset=1 if self.variant=='C1' else 0
        selected=sum(weights[:,i+offset:i+offset+1]*v for i,v in enumerate(branches))
        return x+self.gamma_c*self.protect(u)+self.gamma_s*self.refine(selected), weights

    def forward(self,x):
        return self.forward_with_weights(x)[0]

class D2GLR(nn.Module):
    def __init__(self,variant):
        super().__init__()
        self.project=nn.Conv2d(512,256,1,bias=True)
        self.block=GLR(256,variant)
    def forward(self,x):
        if x.ndim!=4 or x.shape[1:]!=(512,16,16):
            raise ValueError(f'D2 expected [N,512,16,16], received {tuple(x.shape)}')
        return self.block(self.project(x))

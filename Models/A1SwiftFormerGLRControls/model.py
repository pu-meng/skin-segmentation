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

class FullFFN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(channels, 2*channels, 1), nn.GELU(), nn.Conv2d(2*channels, channels, 1))
    def forward(self, x):
        return self.net(x)

class SplitFFN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        if channels % 2 or channels < 2:
            raise ValueError('SplitFFN requires positive even channel count')
        self.ct = channels // 2
        self.net = nn.Sequential(nn.Conv2d(self.ct, 2*self.ct, 1), nn.GELU(), nn.Conv2d(2*self.ct, self.ct, 1))
    def forward(self, x):
        a, b = x.split([self.ct, x.shape[1]-self.ct], dim=1)
        x = torch.cat((self.net(a), b), dim=1)
        n,c,h,w = x.shape
        return x.reshape(n,2,c//2,h,w).transpose(1,2).contiguous().reshape(n,c,h,w)

class GLR(nn.Module):
    def __init__(self, channels, variant):
        super().__init__()
        if variant not in ('CTRL-U','CTRL-AVG'):
            raise ValueError('Only controls are implemented')
        self.variant = variant
        self.pre = nn.Sequential(nn.Conv2d(channels,channels,1,bias=True),dw(channels,3),gn(channels),nn.GELU())
        self.branches = nn.ModuleList([nn.Sequential(dw(channels,3),gn(channels),nn.GELU()),
                                       nn.Sequential(dw(channels,7),gn(channels),nn.GELU()),
                                       nn.Sequential(nn.Conv2d(channels,channels,1,bias=True),gn(channels),nn.GELU())])
        d=max(8,channels//8)
        if variant == 'CTRL-U':
            self.context_dw=dw(channels,3)
            self.context_project=nn.Conv2d(channels,d,1,bias=True)
        self.protect=nn.Sequential(dw(channels,3),gn(channels),nn.GELU(),nn.Conv2d(channels,channels,1,bias=True))
        self.refine = SplitFFN(channels)
        self.gamma_c=nn.Parameter(torch.tensor(.01))
        self.gamma_s=nn.Parameter(torch.tensor(.01))
        if variant == 'CTRL-U':
            self.select_dw=dw(d,3)
            self.select_norm=gn(d)
            self.select_out=nn.Conv2d(d,4,1,bias=True)
            nn.init.zeros_(self.select_out.weight); nn.init.zeros_(self.select_out.bias)

    def forward_with_weights(self,x):
        if min(x.shape[-2:])<4:
            raise ValueError('GLR requires H,W >=4 for fixed context pooling')
        u=self.pre(x)
        branches=[branch(u) for branch in self.branches]
        if self.variant == 'CTRL-U':
            context=self.context_project(self.context_dw(F.adaptive_avg_pool2d(u,(4,4))))
            context=F.interpolate(context,size=u.shape[-2:],mode='bilinear',align_corners=False)
            weights=self.select_out(F.gelu(self.select_norm(self.select_dw(context)))).softmax(dim=1)
        else:
            weights=u.new_full((u.shape[0],4,*u.shape[-2:]),0.25)
        offset=1
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

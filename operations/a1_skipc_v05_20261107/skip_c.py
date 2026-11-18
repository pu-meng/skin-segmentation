"""User Skip1 C v0.5: replace e3 in full; D is condition-only."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class SkipC(nn.Module):
    def __init__(self,c=64,cg=64):
        super().__init__()
        t=torch.arange(-2,3,dtype=torch.float32)
        k=torch.exp(-t.square()/2);k=k/k.sum()
        self.register_buffer('gaussian',(k[:,None]*k[None,:])[None,None].repeat(c,1,1,1))
        self.freq=nn.Linear(c,3*c);self.retain=nn.Linear(c,c)
        nn.init.zeros_(self.freq.weight);nn.init.zeros_(self.freq.bias)
        nn.init.zeros_(self.retain.weight);nn.init.constant_(self.retain.bias,math.log(17.0))
        self.dw=nn.Conv2d(c,c,3,padding=1,groups=c)
        self.pw=nn.Conv2d(c,c,1)
        self.gproj=nn.Conv2d(cg,1,1)
        self.spatial=nn.Conv2d(3,1,3,padding=1)
        self.theta=nn.Parameter(torch.tensor(math.log(math.expm1(.01))))
    def gauss(self,x):
        return F.conv2d(F.pad(x,(2,2,2,2),mode='reflect'),self.gaussian,groups=x.shape[1])
    def components(self,x,g):
        b,c,_,_=x.shape;v=x.mean((2,3))
        l1=self.gauss(x);l2=self.gauss(l1)
        bands=torch.stack([x-l1,l1-l2,l2],2)
        p=self.freq(v).reshape(b,c,3).softmax(2)
        freq=3*(p[:,:,:,None,None]*bands).sum(2)
        keep=(.1+.9*self.retain(v).sigmoid())[:,:,None,None]
        delta=self.pw(self.dw(x))
        ad=self.spatial(torch.cat([x.mean(1,keepdim=True),x.max(1,keepdim=True).values,self.gproj(g)],1)).sigmoid()
        gamma=F.softplus(self.theta)
        return dict(p=p,F=freq,As=keep,Delta=delta,Ad=ad,gamma=gamma)
    def forward(self,x,g):
        z=self.components(x,g)
        return z['As']*z['F']+z['gamma']*z['Ad']*z['Delta'],g

def build_candidate(pretrained=True):
    from source.decoder_dual_a_20261106.candidate import build_candidate as build_a1
    model=build_a1(pretrained=pretrained)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(51);model.decoder.e3=SkipC()
    return model

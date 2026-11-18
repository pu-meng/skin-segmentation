"""CP-DWPO replacements for named D3/D2/D1, leaving D0 and both skips intact."""
import math
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from source.a1_s1_skipv06_bottle_20261110.candidate import RepDW

@dataclass(frozen=True)
class Config:
    enabled: bool = True
    enable_wave: bool = True
    dynamic_wave: bool = True
    enable_eaa: bool = True
    enable_refine: bool = True
    shallow_low_res: bool = True

def basis(n, device, dtype):
    i=torch.arange(n,device=device,dtype=dtype)
    m=torch.cos(math.pi*i[:,None]*(i[None,:]+0.5)/n)*math.sqrt(2/n)
    m[0] /= math.sqrt(2)
    return m

def dct2(x):
    a=basis(x.shape[-2],x.device,x.dtype); b=basis(x.shape[-1],x.device,x.dtype)
    return a @ x @ b.T

def idct2(x):
    a=basis(x.shape[-2],x.device,x.dtype); b=basis(x.shape[-1],x.device,x.dtype)
    return a.T @ x @ b

class Wave(nn.Module):
    def __init__(self,c,dynamic=True):
        super().__init__(); self.dynamic=dynamic; self.raw_speed=nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))
        self.speed_net=nn.Sequential(nn.Linear(c,max(8,c//4)),nn.GELU(),nn.Linear(max(8,c//4),1)) if dynamic else None
    def forward(self,x):
        work=x.float() if x.dtype in (torch.float16,torch.bfloat16) else x
        h,w=work.shape[-2:]; kx=torch.arange(h,device=x.device,dtype=work.dtype)*math.pi/h; ky=torch.arange(w,device=x.device,dtype=work.dtype)*math.pi/w
        radius=torch.sqrt(kx[:,None].square()+ky[None,:].square())
        raw=self.raw_speed + (self.speed_net(work.mean((2,3))).reshape(-1,1,1,1) if self.dynamic else 0)
        speed=F.softplus(raw); resp=torch.cos(speed*radius); return idct2(dct2(work)*resp).to(x.dtype)

class EAA(nn.Module):
    """SwiftFormer EfficientAdditiveAttnetion, one head, equal input/token width."""
    def __init__(self,c):
        super().__init__(); self.to_query=nn.Linear(c,c); self.to_key=nn.Linear(c,c); self.w_g=nn.Parameter(torch.randn(c,1)); self.scale_factor=c**-.5; self.Proj=nn.Linear(c,c); self.final=nn.Linear(c,c)
    def forward(self,x):
        b,c,h,w=x.shape; t=x.flatten(2).transpose(1,2)
        q=F.normalize(self.to_query(t),dim=-1); k=F.normalize(self.to_key(t),dim=-1)
        a=F.normalize((q@self.w_g)*self.scale_factor,dim=1); g=(a*q).sum(dim=1,keepdim=True)
        return self.final(self.Proj(g*k)+q).transpose(1,2).reshape(b,c,h,w)

def dw(c,k=3,stride=1):return nn.Conv2d(c,c,k,stride=stride,padding=k//2,groups=c,bias=False)
class CPDWPOBlock(nn.Module):
    def __init__(self,ci,co,parts,shallow=False,config=Config()):
        super().__init__(); l,f,g=parts; assert l+f+g==co and min(parts)>0
        self.config=config; self.parts=parts; self.shortcut=nn.Identity() if ci==co else nn.Conv2d(ci,co,1,bias=False)
        self.p1=nn.Sequential(nn.Conv2d(ci,co,1),dw(co),nn.GELU(),nn.Conv2d(co,co,1))
        self.pre=RepDW(co,3); self.pre_scale=nn.Parameter(torch.tensor(.01))
        self.local=nn.Sequential(RepDW(l,3),nn.GELU(),dw(l,5),nn.Conv2d(l,l,1))
        self.freq_pre=dw(f,3,2 if shallow and config.shallow_low_res else 1); self.freq_post=dw(f)
        self.wave=Wave(f,config.dynamic_wave); self.freq_gate=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(f,f,1),nn.Sigmoid())
        self.freq_scale=nn.Parameter(torch.tensor(.1))
        self.eaa=EAA(g); self.global_pw=nn.Conv2d(g,g,1)
        self.fuse=nn.Sequential(nn.Conv2d(co,co,1),dw(co))
        self.refine=RepDW(co,3); self.alpha=nn.Parameter(torch.tensor(.01)); self.beta=nn.Parameter(torch.tensor(.01))
        self.gate=nn.Sequential(nn.Conv2d(co,1 if shallow else co,1),nn.Sigmoid())
        self.out=nn.Sequential(RepDW(co,3),nn.Conv2d(co,co,1)); self.out_scale=nn.Parameter(torch.full((1,co,1,1),.01))
    def forward(self,x):
        z=self.p1(x); z=z+self.pre_scale*self.pre(z); l,f,g=z.split(self.parts,1)
        l=l+self.local(l); u=self.freq_pre(f)
        if self.config.enable_wave:u=self.wave(u)
        if u.shape[-2:]!=f.shape[-2:]:u=F.interpolate(u,size=f.shape[-2:],mode='bilinear',align_corners=False)
        f=f+self.freq_scale*self.freq_gate(f)*self.freq_post(u)
        if self.config.enable_eaa:g=g+self.global_pw(self.eaa(g))
        z=self.fuse(torch.cat((l,f,g),1))
        if self.config.enable_refine:
            gate_input=z if self.gate[0].out_channels==1 else z.mean((2,3),keepdim=True)
            z=z+self.alpha*self.refine(z)+self.beta*self.gate(gate_input)*z
        return self.shortcut(x)+self.out_scale*self.out(z)

def build_candidate(pretrained=True,base='skip',config=Config()):
    if base=='skip':
        from source.a1_s1_skip1_v06_20261108.candidate import build_candidate as build
    elif base=='bottle':
        from source.a1_s1_skipv06_bottle_20261110.candidate import build_candidate as build
    else:raise ValueError(base)
    model=build(pretrained=pretrained)
    if config.enabled:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(4711)
            model.decoder.stages[3]=CPDWPOBlock(512,512,(205,154,153),config=config)
            model.decoder.stages[2]=CPDWPOBlock(512,256,(128,77,51),config=config)
            model.decoder.stages[1]=CPDWPOBlock(256,128,(77,38,13),shallow=True,config=config)
    return model

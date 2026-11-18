"""Section 5.2: single-level Laplacian, bidirectional Mamba, three spatial gates."""
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'source/comparison'))
from pretrained_unet2d import PretrainedUNet2D, prepare_imports
from mamba_ssm.modules.mamba_simple import Mamba
from Models.TinyViM.blocks import RepDW

class SpatialGain(nn.Sequential):
    def __init__(self,c=64):
        super().__init__(nn.Conv2d(2*c,32,1),nn.GELU(),nn.Conv2d(32,1,3,padding=1))
        nn.init.zeros_(self[-1].weight)
        nn.init.zeros_(self[-1].bias)
    def forward(self,a,b):
        return 2*torch.sigmoid(super().forward(torch.cat((a,b),dim=1)))

class LaplacianSkip(nn.Module):
    def __init__(self,c=64):
        super().__init__()
        v=torch.tensor([1.,4.,6.,4.,1.])/16
        self.register_buffer('gaussian',(v[:,None]*v[None,:])[None,None])
        self.forward_scan=Mamba(c,d_state=16,d_conv=4,expand=2)
        self.backward_scan=Mamba(c,d_state=16,d_conv=4,expand=2)
        self.context_projection=nn.Conv2d(c,c,1)
        self.detail=nn.Sequential(RepDW(c),nn.GELU(),nn.Conv2d(c,c,1))
        self.gamma_d=nn.Parameter(torch.full((1,c,1,1),.001))
        self.gamma_h=nn.Parameter(torch.full((1,c,1,1),.001))
        self.gate_h=SpatialGain(c);self.gate_l=SpatialGain(c);self.gate_d=SpatialGain(c)

    def smooth(self,x):
        weight=self.gaussian.to(dtype=x.dtype).expand(x.shape[1],1,5,5)
        return F.conv2d(F.pad(x,(2,2,2,2),mode='replicate'),weight,groups=x.shape[1])

    def reduce(self,x):
        return self.smooth(x)[:,:,::2,::2]

    def expand(self,x,size):
        return self.smooth(F.interpolate(x,size=size,mode='bilinear',align_corners=False))

    def context(self,d):
        low=self.reduce(d)
        tokens=low.flatten(2).transpose(1,2)
        f=self.forward_scan(tokens)
        b=self.backward_scan(tokens.flip(1)).flip(1)
        correction=((f+b)*.5).transpose(1,2).reshape_as(low)
        return self.expand(low+self.gamma_d*self.context_projection(correction),d.shape[-2:])

    def forward(self,s,d,return_details=False,identity=False):
        low=self.reduce(s);expanded=self.expand(low,s.shape[-2:]);h=s-expanded
        if identity:
            return expanded+h,d
        hp=h+self.gamma_h*self.detail(h)
        c=self.context(d)
        gh=self.gate_h(c,hp);gl=self.gate_l(expanded,hp);gd=self.gate_d(d,hp)
        selected=gh*hp
        shat=gl*expanded+selected;dhat=gd*d
        if return_details:
            return shat,dhat,dict(low=low,expanded=expanded,h=h,hp=hp,context=c,
                                  gate_h=gh,gate_l=gl,gate_d=gd,selected=selected)
        return shat,dhat

class TinyViMLaplacianSkip(PretrainedUNet2D):
    def __init__(self,pretrained=True):
        prepare_imports('mambavision_t')
        super().__init__('tinyvim_s',pretrained=pretrained,new_layers_seed=42)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(44)
            self.skip_refiner=LaplacianSkip(64)

    def decode_features(self,features,refine=True,identity=False):
        x=features[-1];skips=list(reversed(features[:-1]))
        for i,(up,block) in enumerate(zip(self.decoder.ups,self.decoder.blocks)):
            x=up(x)
            if i<3:
                skip=skips[i]
                if i==2 and refine:
                    skip,x=self.skip_refiner(skip,x,identity=identity)
                x=torch.cat((x,skip),dim=1)
            x=block(x)
        return torch.sigmoid(self.decoder.head(x))

    def forward(self,x):
        features=[layer(f) for layer,f in zip(self.align,self.raw_features(x))]
        return self.decode_features(features)

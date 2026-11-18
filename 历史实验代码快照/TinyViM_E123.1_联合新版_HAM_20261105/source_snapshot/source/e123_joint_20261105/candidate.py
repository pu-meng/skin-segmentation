"""E123.1: original E1 DStage + latest chapter2 E2 + chapter3 dual-source E3."""
from pathlib import Path
import sys,math
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba'),str(ROOT/'source/three_module_impl_20261104')]
import torch
from torch import nn
from torch.nn import functional as F
from source.three_module_impl_20261104.candidate import DStage
from Models.TinyViM import blocks as b

class LN(nn.Module):
 def __init__(self,c):super().__init__();self.norm=nn.LayerNorm(c,eps=1e-5)
 def forward(self,x):return self.norm(x.permute(0,2,3,1)).permute(0,3,1,2).contiguous()
def pw(ci,co):return nn.Conv2d(ci,co,1,bias=True)
def dw(c,k=3,reflect=False):return nn.Conv2d(c,c,k,padding=k//2,groups=c,bias=False,padding_mode='reflect' if reflect else 'zeros')
def surround(ci,co,reflect=False):return nn.Sequential(dw(ci,reflect=reflect),pw(ci,co),LN(co),nn.GELU())
def gate(c,h,out):
 m=nn.Sequential(pw(c,h),nn.GELU(),pw(h,out));nn.init.zeros_(m[-1].weight);nn.init.zeros_(m[-1].bias);return m
class Rep(nn.Module):
 def __init__(self,c,k,reflect=False):
  super().__init__();self.large=nn.Sequential(dw(c,k,reflect),nn.BatchNorm2d(c));self.one=nn.Conv2d(c,c,1,groups=c,bias=True);self.bn=nn.BatchNorm2d(c)
 def forward(self,x):return self.bn(self.large(x)+self.one(x)+x)
class Scan(nn.Module):
 """Exactly four independent state16/rank4 width64 scans, aligned and averaged."""
 def __init__(self):
  super().__init__();c=64;r=4;n=16
  self.xw=nn.Parameter(torch.stack([nn.Linear(c,r+2*n,bias=False).weight for _ in range(4)]))
  ds=[b.SS2D.dt_init(r,c) for _ in range(4)]
  self.dtw=nn.Parameter(torch.stack([d.weight for d in ds]));self.dtb=nn.Parameter(torch.stack([d.bias for d in ds]))
  self.A=b.SS2D.A_log_init(n,c,copies=4);self.D=b.SS2D.D_init(c,copies=4)
 def forward(self,x):
  B,C,H,W=x.shape;L=H*W
  row=x.flatten(2);col=x.transpose(2,3).contiguous().flatten(2)
  xs=torch.stack([row,col,row.flip(-1),col.flip(-1)],dim=1)
  xp=torch.einsum('bkdl,kcd->bkcl',xs,self.xw);dt,bb,cc=torch.split(xp,[4,16,16],dim=2)
  dt=torch.einsum('bkrl,kdr->bkdl',dt,self.dtw)
  y=b.SelectiveScan.apply(xs.reshape(B,-1,L).float(),dt.reshape(B,-1,L).contiguous().float(),-self.A.float().exp(),bb.contiguous().float(),cc.contiguous().float(),self.D.float(),self.dtb.flatten().float(),True,1).reshape(B,4,C,L)
  y=(y[:,0]+y[:,2].flip(-1)+y[:,1].reshape(B,C,W,H).transpose(2,3).reshape(B,C,L)+y[:,3].flip(-1).reshape(B,C,W,H).transpose(2,3).reshape(B,C,L))/4
  return y.reshape(B,C,H,W).to(x.dtype)
class GlobalBranch(nn.Module):
 def __init__(self,coarse):
  super().__init__();self.coarse=coarse;self.norm=LN(64);self.pre=pw(64,64)
  self.a=nn.Conv2d(32,32,(7,1),padding=(3,0),groups=32,bias=False);self.b=nn.Conv2d(32,32,(1,7),padding=(0,3),groups=32,bias=False)
  self.scan=Scan();self.post=nn.Sequential(LN(64),pw(64,64))
 def forward(self,x):
  t=F.avg_pool2d(x,2,2) if self.coarse else x
  a,c=self.pre(self.norm(t)).chunk(2,1);t=self.post(self.scan(F.silu(torch.cat([self.a(a),self.b(c)],1))))
  if self.coarse:t=F.interpolate(t,size=x.shape[-2:],mode='bilinear',align_corners=False)
  return x+t
class FourierDyn(nn.Module):
 def __init__(self):
  super().__init__();self.base=nn.Parameter(torch.empty(32,3,3));nn.init.kaiming_uniform_(self.base.view(32,1,3,3),a=math.sqrt(5));self.generate=gate(32,8,288)
 def kernels(self,x):return self.base[None]+.01*torch.tanh(self.generate(x.mean((2,3),keepdim=True)).reshape(-1,32,3,3))
 def forward(self,x):
  with torch.autocast(x.device.type,enabled=False):
   t=x.float();k=self.kernels(t);h,w=t.shape[-2:];size=(h+2,w+2)
   out=torch.fft.irfft2(torch.fft.rfft2(t,s=size)*torch.fft.rfft2(k,s=size),s=size)[...,1:h+1,1:w+1]
  return out.to(x.dtype)
class Local(nn.Module):
 def __init__(self,c,k):
  super().__init__();self.op=nn.Sequential(LN(c),Rep(c,k) if k else FourierDyn(),nn.GELU(),pw(c,c))
 def forward(self,x):return x+self.op(x)
class E2Mixer(nn.Module):
 def __init__(self):
  super().__init__();self.g1=GlobalBranch(False);self.g2=GlobalBranch(True);self.fg=pw(128,128)
  self.l3=Local(48,3);self.l7=Local(48,7);self.lf=Local(32,0);self.fl=pw(128,128)
  self.rg=nn.Parameter(torch.tensor(math.log(.1/.9)));self.rl=nn.Parameter(torch.tensor(math.log(.1/.9)))
  self.cg=gate(128,16,128);self.cl=gate(128,16,128);self.select=gate(256,32,256);self.out=pw(256,256)
 def forward(self,z):
  g0,l0=z.chunk(2,1);a,c=g0.chunk(2,1);a3,a7,af=l0.split([48,48,32],1)
  g=g0+self.rg.sigmoid()*(self.fg(torch.cat([self.g1(a),self.g2(c)],1))-g0)
  l=l0+self.rl.sigmoid()*(self.fl(torch.cat([self.l3(a3),self.l7(a7),self.lf(af)],1))-l0)
  gh=g*(2*self.cg(l.mean((2,3),keepdim=True)).sigmoid());lh=l*(2*self.cl(g.mean((2,3),keepdim=True)).sigmoid())
  joined=torch.cat([gh,lh],1);w=self.select(joined.mean((2,3),keepdim=True)).reshape(-1,2,128,1,1).softmax(1)
  return self.out(torch.cat([2*w[:,0]*gh,2*w[:,1]*lh],1))
class E2(nn.Module):
 def __init__(self):
  super().__init__();self.pre=surround(512,256);self.n1=LN(256);self.n2=LN(256);self.mixer=E2Mixer()
  # Chapter2 FFN unspecified: retain E2.1/TinyViM exact FFN 4x with its two BN layers, no dropout.
  self.ffn=b.FFN(256,1024,256);self.post=surround(256,512);self.gamma=nn.Parameter(torch.full((1,512,1,1),.01))
 def forward(self,x):
  u=self.pre(x);v=u+self.mixer(self.n1(u));w=v+self.ffn(self.n2(v));return x+self.gamma*self.post(w)
class Lap(nn.Module):
 def __init__(self):
  super().__init__();a=torch.tensor([1.,4.,6.,4.,1.])/16;self.register_buffer('kernel',(a[:,None]*a[None,:]).reshape(1,1,5,5));self.n=LN(24);self.op=nn.Sequential(dw(24,reflect=True),nn.GELU(),pw(24,24))
 def forward(self,x):
  t=self.n(x);lo=F.conv2d(F.pad(t,(2,2,2,2),mode='reflect'),self.kernel.expand(24,1,5,5),groups=24);lo=F.interpolate(F.avg_pool2d(lo,2,2),size=x.shape[-2:],mode='bilinear',align_corners=False)
  return x+self.op(t-lo)
class Haar(nn.Module):
 def __init__(self):super().__init__();self.n=LN(24);self.op=nn.Sequential(dw(96,reflect=True),nn.GELU(),pw(96,96))
 @staticmethod
 def analysis(t):
  a=t[...,::2,::2];b=t[...,::2,1::2];c=t[...,1::2,::2];d=t[...,1::2,1::2]
  return torch.cat([(a+b+c+d)/2,(a-b+c-d)/2,(a+b-c-d)/2,(a-b-c+d)/2],1)
 @staticmethod
 def synthesis(t):
  ll,lh,hl,hh=t.chunk(4,1);a=(ll+lh+hl+hh)/2;b=(ll-lh+hl-hh)/2;c=(ll+lh-hl-hh)/2;d=(ll-lh-hl+hh)/2
  return torch.stack([torch.stack([a,b],-1),torch.stack([c,d],-1)],-3).reshape(t.shape[0],t.shape[1]//4,t.shape[2]*2,t.shape[3]*2)
 def forward(self,x):return x+self.synthesis(self.op(self.analysis(self.n(x))))
class FourierE3(nn.Module):
 def __init__(self):
  super().__init__();self.n=LN(16);self.a=nn.Parameter(torch.zeros(16,4));self.out=nn.Sequential(nn.GELU(),pw(16,16))
  y=torch.fft.fftfreq(64)[:,None];x=torch.fft.rfftfreq(64)[None,:];rho=(x*x+y*y).sqrt();mu=torch.tensor([0.,.2,.4,.65]);self.register_buffer('phi',torch.exp(-(rho[None]-mu[:,None,None]).square()/(2*.15**2)))
 def forward(self,x):
  t=self.n(x)
  with torch.autocast(x.device.type,enabled=False):
   m=1+.5*torch.tanh(torch.einsum('cj,jhw->chw',self.a,self.phi));y=torch.fft.irfft2(m[None]*torch.fft.rfft2(t.float(),norm='ortho'),s=(64,64),norm='ortho')
  return x+self.out(y.to(t.dtype))
class Axis(nn.Module):
 def __init__(self):
  super().__init__();self.n=LN(24);self.h=nn.Conv2d(12,12,(1,7),padding=(0,3),groups=12,bias=False,padding_mode='reflect');self.v=nn.Conv2d(12,12,(7,1),padding=(3,0),groups=12,bias=False,padding_mode='reflect');self.out=nn.Sequential(nn.GELU(),pw(24,24))
 def forward(self,x):
  a,b=self.n(x).chunk(2,1);return x+self.out(torch.cat([self.h(a),self.v(b)],1))
class E3Generator(nn.Module):
 def __init__(self):
  super().__init__();self.lap=Lap();self.haar=Haar();self.fourier=FourierE3();self.fuse=pw(64,64);self.n=LN(64)
  self.rep=nn.Sequential(LN(24),Rep(24,3,True),nn.GELU(),pw(24,24));self.axis=Axis();self.coarse=nn.Sequential(LN(16),Rep(16,7,True),nn.GELU(),pw(16,16));self.select=gate(64,16,3);self.out=pw(64,64)
 def forward(self,z):
  a,b,c=z.split([24,24,16],1);t=self.n(self.fuse(torch.cat([self.lap(a),self.haar(b),self.fourier(c)],1)));a,b,c=t.split([24,24,16],1)
  a=a+self.rep(a);b=self.axis(b);c=c+F.interpolate(self.coarse(F.avg_pool2d(c,4,4)),size=c.shape[-2:],mode='bilinear',align_corners=False)
  joined=torch.cat([a,b,c],1);w=self.select(joined.mean((2,3),keepdim=True)).softmax(1)
  return self.out(torch.cat([3*w[:,0:1]*a,3*w[:,1:2]*b,3*w[:,2:3]*c],1))
class E3Side(nn.Module):
 def __init__(self):
  super().__init__();self.pre=surround(64,64,True);self.n1=LN(64);self.gen=E3Generator();self.mix=pw(64,64);self.n2=LN(64)
  self.ffn=nn.Sequential(pw(64,128),nn.GELU(),dw(128,reflect=True),nn.GELU(),pw(128,64));self.post=surround(64,64,True);self.gamma=nn.Parameter(torch.full((1,64,1,1),.01))
 def finish(self,x,u,k):
  h=u+self.mix(k);t=h+self.ffn(self.n2(h));return x+self.gamma*self.post(t)
class E3(nn.Module):
 def __init__(self):super().__init__();self.s=E3Side();self.d=E3Side();self.gs=gate(64,16,64);self.gd=gate(64,16,64)
 def forward(self,s,d):
  us=self.s.pre(s);ud=self.d.pre(d);k=self.s.gen(self.s.n1(us));v=self.d.gen(self.d.n1(ud))
  kh=k*(2*self.gs(v.mean((2,3),keepdim=True)).sigmoid());vh=v*(2*self.gd(k.mean((2,3),keepdim=True)).sigmoid())
  return self.s.finish(s,us,kh),self.d.finish(d,ud,vh)
class Decoder(nn.Module):
 def __init__(self,old):
  super().__init__();self.stages=old.stages;self.ups=old.ups;self.head=old.head
  with torch.random.fork_rng(devices=[]):torch.manual_seed(42);self.stages[3]=DStage()
  self.e2=E2();self.e3=E3()
 def forward(self,f):
  x=self.stages[3](self.e2(f[3]))
  for level in (2,1,0):
   x=self.ups[level](x);s=f[level]
   if level==0:s,x=self.e3(s,x)
   x=self.stages[level](torch.cat([x,s],1))
  return torch.sigmoid(F.interpolate(self.head(x),size=(256,256),mode='bilinear',align_corners=False))
def build_candidate(pretrained=True):
 from Models.TinyViM.four_stage_decoder import build_four_stage_baseline
 model=build_four_stage_baseline(pretrained=pretrained)
 with torch.random.fork_rng(devices=[]):torch.manual_seed(42);model.decoder=Decoder(model.decoder)
 return model

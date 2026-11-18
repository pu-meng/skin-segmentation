"""Independent E2/E3; no changes to E1 or official backbone."""
from pathlib import Path
import sys,math,importlib.util
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
class Gate(nn.Module):
 def __init__(self,source,target):
  super().__init__();self.proj=nn.Conv2d(source,target,1);self.raw=nn.Parameter(torch.zeros(1,target,1,1))
 def forward(self,a,b):return a*(1+torch.tanh(self.raw)*torch.sigmoid(self.proj(b)))

def conv(c,k=1,groups=1,dilation=1):
 return nn.Sequential(nn.Conv2d(c,c,k,padding=dilation*(k//2),groups=groups,dilation=dilation,bias=False),nn.BatchNorm2d(c),nn.GELU())
def axial(c):
 return nn.Sequential(nn.Conv2d(c,c,(1,7),padding=(0,3),groups=c,bias=False),nn.BatchNorm2d(c),nn.GELU(),nn.Conv2d(c,c,(7,1),padding=(3,0),groups=c,bias=False),nn.BatchNorm2d(c),nn.GELU())

class FullScan(nn.Module):
 """Full-channel four-way scan; no TinyViM alpha split/pool."""
 def __init__(self,c):
  super().__init__()
  from Models.TinyViM import blocks as b
  self.input=b.Conv2d_BN(c,c,1);self.local=nn.Sequential(nn.Conv2d(c,c,3,padding=1,groups=c,bias=True),nn.SiLU())
  r=math.ceil(c/16);state=16
  self.x_proj_weight=nn.Parameter(torch.stack([nn.Linear(c,r+2*state,bias=False).weight for _ in range(4)]))
  dt=[b.SS2D.dt_init(r,c) for _ in range(4)]
  self.dt_weight=nn.Parameter(torch.stack([t.weight for t in dt]));self.dt_bias=nn.Parameter(torch.stack([t.bias for t in dt]))
  self.A_logs=b.SS2D.A_log_init(state,c,copies=4);self.Ds=b.SS2D.D_init(c,copies=4)
  self.norm=nn.LayerNorm(c);self.output=b.Conv2d_BN(c,c,1)
 def forward(self,x):
  from Models.TinyViM import blocks as b
  x=self.local(self.input(x))
  y=b.cross_selective_scan(x,self.x_proj_weight,None,self.dt_weight,self.dt_bias,self.A_logs,self.Ds,self.norm,nrows=1,delta_softplus=True,force_fp32=self.training)
  return self.output(y.permute(0,3,1,2).contiguous())

def ideal_split(x):
 with torch.autocast(x.device.type,enabled=False):
  h,w=x.shape[-2:];ky=torch.fft.fftfreq(h,device=x.device);kx=torch.fft.rfftfreq(w,device=x.device)
  mask=(ky[:,None].square()+kx[None,:].square()<=.25**2)
  f=torch.fft.rfft2(x.float(),norm='ortho')
  lo=torch.fft.irfft2(f*mask,s=(h,w),norm='ortho');hi=torch.fft.irfft2(f*(~mask),s=(h,w),norm='ortho')
  return lo,hi

def skip_split(x):
 c=x.shape[1];a=x.new_tensor([1.,4.,6.,4.,1.]);kernel=(a[:,None]*a[None,:]/256).view(1,1,5,5).expand(c,1,5,5)
 blur=F.conv2d(F.pad(x,(2,2,2,2),mode='replicate'),kernel,groups=c)
 lo=F.interpolate(F.avg_pool2d(blur,2,2),size=x.shape[-2:],mode='bilinear',align_corners=False)
 return lo,x-lo

class Triple(nn.Module):
 def __init__(self,low):
  super().__init__()
  from Models.TinyViM.blocks import RepDW
  self.widths=(172,170,170)
  self.project=nn.Conv2d(512,512,1)
  self.branches=nn.ModuleList([FullScan(172) if low else RepDW(172),RepDW(170) if low else axial(170),axial(170) if low else conv(170,3,170,2)])
  self.gates=nn.ModuleList([Gate(170,172),Gate(170,170),Gate(172,170)])
  self.fuse=nn.Conv2d(512,512,1)
 def forward(self,x):
  a,b,c=[m(t) for m,t in zip(self.branches,self.project(x).split(self.widths,dim=1))]
  return self.fuse(torch.cat([self.gates[0](a,b),self.gates[1](b,c),self.gates[2](c,a)],1))

class MultiDW(nn.Module):
 """Official four dilation combinations adapted explicitly to depthwise."""
 def __init__(self,c):
  super().__init__();self.d1=nn.Conv2d(c,c,3,padding=1,groups=c);self.d3=nn.Conv2d(c,c,3,padding=3,dilation=3,groups=c);self.d5=nn.Conv2d(c,c,3,padding=5,dilation=5,groups=c);self.p=nn.Conv2d(c,c,1)
 def forward(self,x):
  return F.relu(self.d1(x))+F.relu(self.p(self.d3(x)))+F.relu(self.p(self.d3(self.d1(x))))+F.relu(self.p(self.d5(self.d3(self.d1(x)))))

def official_attention():
 # Load untouched official package under an isolated name, not a global Models alias.
 name='_e23_surmt';p=ROOT/'references/code/SU-RMT/Models/SURMT'
 if name+'.HSAModule' not in sys.modules:
  import types
  package=types.ModuleType(name);package.__path__=[str(p)];sys.modules[name]=package
  # Exact original AST for these classes/helpers; omit unrelated pywt-dependent imports.
  import ast
  text=(p/'HSAModule.py').read_text();tree=ast.parse(text)
  nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ('rotate_every_two','theta_shift','RoPE2D','FourierAttention')]
  m=types.ModuleType(name+'.HSAModule');m.__dict__.update(torch=torch,nn=nn,F=F)
  exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p/'HSAModule.py'),'exec'),m.__dict__);sys.modules[m.__name__]=m
 return sys.modules[name+'.HSAModule']

class Attention(nn.Module):
 def __init__(self,high):
  super().__init__();self.high=high;self.q=nn.Conv2d(512,256,1);self.k=nn.Conv2d(512,256,1);self.v=nn.Conv2d(512,256,1);self.out=nn.Conv2d(256,512,1)
  if high:
   official=official_attention();self.kfeat=MultiDW(256);self.vfeat=MultiDW(256)
   self.core=official.FourierAttention(4,64,64);self.rope=official.RoPE2D(256,4);self.norm=nn.LayerNorm(256)
   self.context=nn.Sequential(nn.Linear(256,256),nn.GELU(),nn.Linear(256,4))
 def forward(self,x):
  b,_,h,w=x.shape;q=self.q(x);k=self.k(x);v=self.v(x)
  if self.high:
   with torch.autocast(x.device.type,enabled=False):
    q=q.float();k=self.kfeat(k.float());v=self.vfeat(v.float())
    heads=lambda t:t.flatten(2).transpose(1,2).reshape(b,h*w,4,64).permute(0,2,1,3)
    context=self.norm(k.flatten(2).transpose(1,2)).mean(1);weight=torch.sigmoid(self.context(context)).view(b,4,1)
    sin,cos=self.rope(h,w,x.device);o=self.core(heads(q),heads(k),heads(v),weight,h,w,sin,cos).permute(0,2,1,3).reshape(b,h*w,256).transpose(1,2).reshape(b,256,h,w)
  else:
   pool=lambda t:torch.cat([F.max_pool2d(t,s,s).flatten(2) for s in (2,3,5,6)],dim=2)
   q=q.flatten(2).transpose(1,2).reshape(b,h*w,4,64).permute(0,2,1,3)
   k=pool(k).transpose(1,2).reshape(b,-1,4,64).permute(0,2,1,3);v=pool(v).transpose(1,2).reshape(b,-1,4,64).permute(0,2,1,3)
   o=((q@k.transpose(-1,-2)*(.125)).softmax(-1)@v).permute(0,2,1,3).reshape(b,h*w,256).transpose(1,2).reshape(b,256,h,w)
  return self.out(o)

class HSAMixer(nn.Module):
 def __init__(self):
  super().__init__();self.low=Triple(True);self.high=Triple(False);self.gl=Gate(512,512);self.gh=Gate(512,512);self.al=Attention(False);self.ah=Attention(True);self.fuse=nn.Conv2d(1024,512,1)
 def forward(self,x):
  l,h=ideal_split(x);l=self.low(l);h=self.high(h)
  return self.fuse(torch.cat([self.al(self.gl(l,h)),self.ah(self.gh(h,l))],1))

class SkipMixer(nn.Module):
 def __init__(self):
  super().__init__()
  from Models.TinyViM.blocks import RepDW
  self.low=nn.Sequential(nn.Conv2d(64,64,1),FullScan(64));self.h1=RepDW(64);self.h2=axial(64);self.hf=nn.Conv2d(128,64,1);self.gl=Gate(64,64);self.gh=Gate(64,64);self.fuse=nn.Conv2d(128,64,1)
 def forward(self,x):
  l,h=skip_split(x);l=self.low(l);h=self.hf(torch.cat([self.h1(h),self.h2(h)],1))
  return self.fuse(torch.cat([self.gl(l,h),self.gh(h,l)],1))

class CompleteModule(nn.Module):
 def __init__(self,c,mixer):
  super().__init__()
  from Models.TinyViM.blocks import FFN
  self.pre=conv(c,3);self.mixer=mixer;self.ffn=FFN(c,c*4,c);self.post=conv(c,3)
 def forward(self,x):
  x=self.pre(x);x=x+self.mixer(x);x=x+self.ffn(x);return self.post(x)

class DecoderWithModule(nn.Module):
 def __init__(self,decoder,arm):
  super().__init__();self.stages=decoder.stages;self.ups=decoder.ups;self.head=decoder.head;self.arm=arm
  self.enhance=CompleteModule(512,HSAMixer()) if arm=='E2' else CompleteModule(64,SkipMixer())
 def forward(self,features):
  f=list(features);idx=3 if self.arm=='E2' else 0;f[idx]=self.enhance(f[idx])
  x=self.stages[3](f[3])
  for level in (2,1,0):x=self.stages[level](torch.cat([self.ups[level](x),f[level]],1))
  return torch.sigmoid(F.interpolate(self.head(x),size=(256,256),mode='bilinear',align_corners=False))

def build_candidate(pretrained=True,arm=None):
 from Models.TinyViM.four_stage_decoder import build_four_stage_baseline
 if arm not in (None,'E2','E3'):raise ValueError(arm)
 model=build_four_stage_baseline(pretrained=pretrained)
 if arm:
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(42);model.decoder=DecoderWithModule(model.decoder,arm)
 return model

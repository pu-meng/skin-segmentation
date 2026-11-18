from pathlib import Path
import os,sys,json,io,importlib,copy
MODE=sys.argv[1]
if MODE=='cpu':os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
import torch
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t');prepare_imports('tinyvim_s')
from candidate import *
from selection import RUN,CHOICES
HERE=Path(__file__).parent
if MODE=='cpu':
 from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
 class Reference:
  @staticmethod
  def apply(u,delta,A,B,C,D=None,delta_bias=None,delta_softplus=False,nrows=1):
   assert u.device.type=='cpu';return selective_scan_ref(u,delta,A,B,C,D,z=None,delta_bias=delta_bias,delta_softplus=delta_softplus)
 for name in ['Models.TinyViM.blocks','model.tvimblock']:importlib.import_module(name).SelectiveScan=Reference
checks={}
def check(k,v):
 assert v,k
 checks[k]=True;print('PASS',k,flush=True)
torch.set_num_threads(4);torch.manual_seed(42);device=torch.device('cpu' if MODE=='cpu' else 'cuda:0')
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
sc=Scan().to(device);x=torch.randn(1,64,4,6,device=device,requires_grad=True)
if MODE=='gpu':torch.cuda.reset_peak_memory_stats()
y=sc(x);seq=[x.flatten(2),x.transpose(2,3).flatten(2)];seq+=[t.flip(-1) for t in seq];outs=[]
for i,t in enumerate(seq):
 xp=F.conv1d(t,sc.xw[i][:,:,None]);dt,bb,cc=xp.split([4,16,16],1);dt=F.conv1d(dt,sc.dtw[i][:,:,None])
 z=selective_scan_ref(t,dt,-sc.A[i*64:(i+1)*64].exp(),bb,cc,sc.D[i*64:(i+1)*64],delta_bias=sc.dtb[i],delta_softplus=True)
 if i>=2:z=z.flip(-1)
 if i%2:z=z.reshape(1,64,6,4).transpose(2,3).reshape(1,64,24)
 outs.append(z)
ref=torch.stack(outs).mean(0).reshape_as(x);print('NUMERICAL_OUTPUT',float((y-ref).abs().max()),float((y-ref).abs().mean()),float(ref.abs().max()),flush=True)
g1=torch.autograd.grad(y.square().mean(),x,retain_graph=True)[0];g2=torch.autograd.grad(ref.square().mean(),x)[0];print('NUMERICAL_GRAD',float((g1-g2).abs().max()),float((g1-g2).abs().mean()),float(g2.abs().max()),flush=True)

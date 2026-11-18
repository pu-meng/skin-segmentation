"""CPU checks of the exact small source components used in the RWKV diagrams."""
from pathlib import Path
import ast,json,hashlib,subprocess
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[3]
REF=ROOT/'references/code'
OUT=ROOT/'references/核对记录/rwkv_20261103'
OUT.mkdir(parents=True,exist_ok=True)

def extract(path,names):
 tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in names]
 assert len(nodes)==len(names)
 ns={'torch':torch,'nn':nn,'F':F}
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
 return ns
checks={}
q=extract(REF/'RWKV-UNet/rwkv_unet.py',['q_shift'])['q_shift']
x=torch.arange(1*12*8).reshape(1,12,8).float()
y=q(x,patch_resolution=(3,4));a=x.transpose(1,2).reshape(1,8,3,4);b=y.transpose(1,2).reshape_as(a)
assert torch.equal(b[:,:2,:,1:],a[:,:2,:,:-1])and not b[:,:2,:,0].any()
assert torch.equal(b[:,2:4,:,:-1],a[:,2:4,:,1:])and not b[:,2:4,:,-1].any()
assert torch.equal(b[:,4:6,1:,:],a[:,4:6,:-1,:])and not b[:,4:6,0,:].any()
assert torch.equal(b[:,6:8,:-1,:],a[:,6:8,1:,:])and not b[:,6:8,-1,:].any()
checks['q_shift']='four channel groups shift right/left/down/up; zero boundary; non-square 3x4 checked'
ema=extract(REF/'URWKV/model/modules/urwkv.py',['MultiStateTokenShift'])['MultiStateTokenShift']()
a=torch.full((1,4,8),8.);hist=[torch.full_like(a,2),torch.full_like(a,4)]
z=ema.dynamicInterpolate(a,hist);assert torch.all(z==4.5)
checks['ema_actual_order']={'current':8,'oldest_to_newest':[2,4],'actual_fold':float(z[0,0,0]),'single_recursive_read_of_latest':6,'alpha':.5}
lan=extract(REF/'URWKV/model/modules/LAN.py',['LuminanceAdaptiveNorm'])['LuminanceAdaptiveNorm'](8).eval()
x=torch.randn(1,16,8);past=[torch.randn(1,16,2,2),torch.randn(1,8,4,4)]
keys_before=set(lan.state_dict());y1=lan(x,past,(4,4));y2=lan(x,past,(4,4));assert not torch.equal(y1,y2)
assert keys_before==set(lan.state_dict());assert not any('conv' in key for key in keys_before)
checks['lan_dynamic_layers']={'same_input_eval_max_difference':float((y1-y2).abs().max()),'state_dict_keys':sorted(keys_before),'finding':'aggregation Conv2d/Linear are constructed inside forward and not registered; repeated eval differs'}
ssf=extract(REF/'URWKV/model/modules/SSF.py',['InceptionModule','SSF'])['SSF'](3,[16,8,4],4)
features=[torch.randn(1,4,8,8),torch.randn(1,8,4,4),torch.randn(1,16,2,2)]
x=torch.randn(1,4,8,8,requires_grad=True);out=ssf(x,features[:]);assert out.shape==x.shape;out.sum().backward()
assert all(p.grad is None for p in ssf.align_convs.parameters())
checks['ssf']={'input_shape':list(x.shape),'output_shape':list(out.shape),'align_convs_called':False,'full_channel_skip':'only matching-resolution encoder feature; three maps predict a one-channel spatial gate'}
# Paper Eq16: dense causal weighted average and the prefix-state recurrence agree.
torch.manual_seed(1);k=torch.randn(2,5,3,dtype=torch.float64);v=torch.randn_like(k);w=torch.rand(3,dtype=torch.float64);u=torch.randn(3,dtype=torch.float64)
a=torch.zeros(2,3,dtype=torch.float64);b=torch.zeros_like(a);errors=[]
for t in range(5):
 e=(u+k[:,t]).exp();rec=(a+e*v[:,t])/(b+e)
 terms=[(-(t-1-i)*w+k[:,i]).exp()for i in range(t)]
 den=sum(terms,torch.zeros_like(a))+e;num=sum([z*v[:,i]for i,z in enumerate(terms)],torch.zeros_like(a))+e*v[:,t]
 errors.append(float((rec-num/den).abs().max()));a=(-w).exp()*a+k[:,t].exp()*v[:,t];b=(-w).exp()*b+k[:,t].exp()
assert max(errors)<1e-12;checks['rwkv4_eq16_dense_vs_recurrence_max_error']=max(errors)
ccm_ns=extract(REF/'RWKV-UNet/ccm/ccm.py',['VRWKV_ChannelMix','CCMix'])
ccm_ns['DropPath']=nn.Identity
try:
 ccm_ns['CCMix']([96,48,32],32,128)
except NameError as error:
 assert "drop_path" in str(error)
 checks['ccm_public_constructor_issue']=str(error)
else:raise AssertionError('expected public constructor to reference undefined drop_path')
# MMCV v1.7.1 default corner padding preserves H,W for patch kernel3/stride1.
dep=OUT/'mmcv_1_7_1_transformer.py'
if dep.exists():
 import math
 tree=ast.parse(dep.read_text());nodes=[n for n in tree.body if isinstance(n,ast.ClassDef)and n.name in ['AdaptivePadding','PatchEmbed']]
 class Base(nn.Module):
  def __init__(self,init_cfg=None):super().__init__()
 ns={'torch':torch,'nn':nn,'F':F,'math':math,'BaseModule':Base,'to_2tuple':lambda x:(x,x)if isinstance(x,int)else x,'build_conv_layer':lambda cfg,*args,**kwargs:nn.Conv2d(*args,**kwargs),'build_norm_layer':None}
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(dep),'exec'),ns)
 pe=ns['PatchEmbed'](in_channels=8,embed_dims=8,kernel_size=3,stride=1,input_size=8)
 result,res=pe(torch.randn(1,8,8,8));assert res==(8,8)and result.shape==(1,64,8)
 checks['mmcv_patch_embed']='source forward with lightweight constructor adapters: kernel3 stride1 default corner padding preserves H,W'
# Static evidence: no edits/imports of full CUDA-dependent networks.
source_paths=[REF/'RWKV-4/model.py',REF/'RWKV-UNet/rwkv_unet.py',REF/'RWKV-UNet/ccm/ccm.py',REF/'RWKV-UNet/cuda/wkv_cuda.cu']+list((REF/'URWKV/model').rglob('*.py'))+[REF/'URWKV/model/modules/cuda/wkv_cuda.cu']
sources={str(p.relative_to(ROOT)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}for p in source_paths}
pdfs={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path('/home/PuMengYu/RWKV.pdf'),Path('/home/PuMengYu/RWKV-UNet.pdf'),Path('/home/PuMengYu/URWKV.pdf')]}
commits={name:subprocess.check_output(['git','-C',str(REF/name),'rev-parse','HEAD'],text=True).strip()for name in ['RWKV-UNet','URWKV']}
commits['RWKV-4']=json.loads((REF/'RWKV-4/source.json').read_text())['commit']
report={'checks':checks,'commits':commits,'pdf_sha256':pdfs,'sources':sources,'scope':'CPU component checks and static forward tracing only; no full-network CUDA execution, training, or test evaluation'}
(OUT/'source_and_cpu_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(checks,ensure_ascii=False,indent=2))

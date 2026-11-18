from pathlib import Path
import hashlib,json,ast
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
S=ROOT/'source/wpo_mr_20261105';O=ROOT/'operations/wpo_mr_20261105';Q=ROOT/'log/TinyViM_B-WPO-MR_HAM_20261105';R=Q/'B-WPO-MR'
assert not S.exists() and not O.exists() and not Q.exists(),'refuse overwrite'
for p in [S,O,R]:p.mkdir(parents=True)
protected=list((ROOT/'source/wpo_skips_20261105').glob('*.*'))+list((ROOT/'source/bd_controls_20261105').glob('*.*'))+list((ROOT/'source/three_module_impl_20261104').glob('*.py'))+list((ROOT/'operations/wpo_skips_20261105').glob('*.*'))
protected += list((ROOT/'log/TinyViM_WPO单Skip位置_HAM_20261105').glob('*/protocol.json'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(O/'before_protection.json').write_text(json.dumps({str(p):sha(p) for p in protected if p.is_file()},indent=2))
def put(name,text):
 p=S/name;p.write_text(text)
 if p.suffix=='.py':ast.parse(text)
put('__init__.py','')
old=ROOT/'source/wpo_skips_20261105'
e1=(ROOT/'source/three_module_impl_20261104/candidate.py').read_text().split('\ndef build_candidate(')[0]
e1=e1.replace('from official_wave import WaveOperator, _fft2d','from source.wpo_mr_20261105.official_wave import WaveOperator, _fft2d')
put('e1_wave.py',e1)
put('official_wave.py',(ROOT/'source/three_module_impl_20261104/official_wave.py').read_text())
put('selection.py', '''from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
ARM='B-WPO-MR';QUEUE=ROOT/'log/TinyViM_B-WPO-MR_HAM_20261105';RUN=QUEUE/ARM
CHOICES=dict(base='E0 reused, not retrained',placement='Align3 -> parallel WPO/split-Mamba-RepDW -> original D3 DoubleConv -> unchanged decoder',equation='A=DStage_E1(X); B=cat(MambaResidual(X[:256]),RepDW(X[256:])); H=X+P(c*cat(A,B)); Y=GELU(BN(PW512(DW3(H))))',input=[512,8,8],split=[256,256],static_c='1024 free trainable scalars, shape1x1024x1x1, init1, shared across batch and spatial, not input generated, no softmax',WPO='exact original E1 DirectionalWaveOperator and DStage, t1 K1 alpha8x5x1 invsoftplus(.05), gamma perchannel .001; internal SiLU z retained',Mamba='official Mamba-1 d_model256 d_state16 d_conv4 expand2, row-major single direction, token t+mixer(LN(t)); internal gate retained',RepDW='project Models/TinyViM/blocks.py RepDW(256), DW3+DW1+identity then BN',projection='Conv1x1 1024to512 standard torch init, bias=True',post='DW3 512to512 groups512 pad1 biasFalse -> PW1 512to512 biasFalse -> BN512 -> GELU; after outer X residual, no third outer residual',defaults='split256/256, c init1, ordinary P init, post convolution are first-version engineering choices, not user-specified hyperparameters',excluded='no cross-branch gate, SE, softmax, new self gate, E1 replacement of D3, E2/E3, ISIC or multiseed',initialization='ImageNet encoder, paired E0 decoder42/align43; additional module isolated seed42',scope='HAM seed42 full network FP32 batch8 256x256 300 epochs; earliest min-val-loss best then fixed 2020-image test',gpu='GPU0 shared maximum two experiment slots; S0/S1 completion before launch, S2 reserved slot retained, use second slot only when S2 really active or fully complete')
''')
put('candidate.py','''"""B-WPO-MR: parallel full-channel E1 and split Mamba/RepDW, static channel scaling."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
import torch
from torch import nn
from torch.nn import functional as F
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t');prepare_imports('tinyvim_s')
from mamba_ssm.modules.mamba_simple import Mamba
from Models.TinyViM.blocks import RepDW
from source.wpo_mr_20261105.e1_wave import DStage
class SplitMambaRepDW(nn.Module):
 def __init__(self):
  super().__init__();self.norm=nn.LayerNorm(256);self.mixer=Mamba(d_model=256,d_state=16,d_conv=4,expand=2);self.local=RepDW(256)
 def forward(self,x):
  a,b=x.split(256,dim=1);n,c,h,w=a.shape;t=a.flatten(2).transpose(1,2);a=(t+self.mixer(self.norm(t))).transpose(1,2).reshape(n,c,h,w)
  return torch.cat([a,self.local(b)],dim=1)
class ParallelBottleneck(nn.Module):
 def __init__(self):
  super().__init__();self.a=DStage();self.b=SplitMambaRepDW();self.c=nn.Parameter(torch.ones(1,1024,1,1));self.projection=nn.Conv2d(1024,512,1)
  self.post=nn.Sequential(nn.Conv2d(512,512,3,padding=1,groups=512,bias=False),nn.Conv2d(512,512,1,bias=False),nn.BatchNorm2d(512),nn.GELU())
 def forward(self,x):
  assert tuple(x.shape[1:])==(512,8,8)
  a=self.a(x);b=self.b(x);h=x+self.projection(self.c*torch.cat([a,b],dim=1));return self.post(h)
class Decoder(nn.Module):
 def __init__(self,old):
  super().__init__();self.stages=old.stages;self.ups=old.ups;self.head=old.head;self.enhance=ParallelBottleneck()
 def forward(self,f):
  assert [tuple(v.shape[1:]) for v in f]==[(64,64,64),(128,32,32),(256,16,16),(512,8,8)]
  x=self.stages[3](self.enhance(f[3]))
  for level in (2,1,0):x=self.stages[level](torch.cat([self.ups[level](x),f[level]],1))
  return torch.sigmoid(F.interpolate(self.head(x),size=(256,256),mode='bilinear',align_corners=False))
def build_candidate(pretrained=True,enabled=True):
 from Models.TinyViM.four_stage_decoder import build_four_stage_baseline
 model=build_four_stage_baseline(pretrained=pretrained)
 if enabled:
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(42);model.decoder=Decoder(model.decoder)
 return model
def added_parameters(model):
 for n,p in model.named_parameters():
  if n.startswith('decoder.enhance.'):yield n,p
''')
exp=(old/'experiment.py').read_text().replace('source.wpo_skips_20261105','source.wpo_mr_20261105')
exp=exp.replace("'E0 four-stage baseline, isolated single-skip WPO location controls'","'E0 four-stage baseline, isolated parallel E1 WPO and split Mamba/RepDW'")
exp=exp.replace("'Single aligned skip WPO; original D3 retained; E1 reused only as non-equal-parameter reference; no E2/E3/RAP/HAM weights'","'Parallel full-channel original E1 DStage and split Mamba/RepDW; original D3 retained; no E2/E3/HAM weights; not initial identity'")
exp=exp.replace("for relative in ['Models/TinyViM/blocks.py']:","for relative in ['Models/TinyViM/blocks.py','source/three_module_impl_20261104/candidate.py','source/three_module_impl_20261104/official_wave.py']:")
exp=exp.replace("report['all_added_parameter_tensors_updated']=True", "report['all_added_parameter_tensors_updated']=True\n   assert model.decoder.enhance.c.numel()==1024 and not torch.equal(model.decoder.enhance.c,before['decoder.enhance.c'])\n   report['static_c_1024_updated']=True")
put('experiment.py',exp)
entry=(old/'entry.py').read_text().replace('source.wpo_skips_20261105','source.wpo_mr_20261105')
put('entry.py',entry)
results=(old/'results.py').read_text().replace('source.wpo_skips_20261105','source.wpo_mr_20261105').replace("labels = {k:k for k in ['WPO-S0','WPO-S1','WPO-S2']}","labels = {ARM:ARM}").replace("仅{ARM.replace('WPO-', '')}追加WPO","瓶颈E1并行split-Mamba/RepDW")
put('results.py',results)
check=(old/'check.py').read_text().replace('source.wpo_skips_20261105','source.wpo_mr_20261105').replace('build_candidate,added_parameters,SkipWave','build_candidate,added_parameters,ParallelBottleneck').replace('ARM,RUN,LEVEL','ARM,RUN')
start=check.index('blocks=[m for m in model.modules()')
end=check.index('x=torch.rand(1,3,256,256)')
check=check[:start]+'''blocks=[m for m in model.modules() if isinstance(m,ParallelBottleneck)]
ck('one_parallel_module',len(blocks)==1)
module=blocks[0];wave=module.a
ck('stage_parameters',list(wave.alpha_embed.shape)==[8,5,1] and list(wave.gamma.shape)==[1,512,1,1] and wave.wave.direction_raw.numel()==512)
ck('gamma_init',torch.equal(wave.gamma.detach(),torch.full_like(wave.gamma,.001)))
ck('fixed_time',not any(n in ['t_raw','t_ch'] for n,p in wave.wave.named_parameters()))
ck('static_c_1024_init1',module.c.numel()==1024 and tuple(module.c.shape)==(1,1024,1,1) and torch.equal(module.c,torch.ones_like(module.c)))
from source.wpo_mr_20261105.e1_wave import DStage,DirectionalWaveOperator
import ast
original=ast.parse((ROOT/'source/three_module_impl_20261104/candidate.py').read_text());local=ast.parse((Path(__file__).parent/'e1_wave.py').read_text())
ck('original_E1_class_AST_exact',all(ast.dump(next(v for v in original.body if isinstance(v,ast.ClassDef) and v.name==n))==ast.dump(next(v for v in local.body if isinstance(v,ast.ClassDef) and v.name==n)) for n in ['DStage','DirectionalWaveOperator']))
ck('original_official_wave_bytes_exact',(ROOT/'source/three_module_impl_20261104/official_wave.py').read_bytes()==(Path(__file__).parent/'official_wave.py').read_bytes())
ck('D3_retained_DoubleConv',model.decoder.stages[3].__class__.__name__=='DoubleConv')
seen={};hooks=[]
for name,sub in [('A',module.a),('B',module.b),('P',module.projection),('post',module.post),('D3',model.decoder.stages[3])]:
 hooks.append(sub.register_forward_hook(lambda m,i,o,n=name:seen.update({n:dict(input=list(i[0].shape),output=list(o.shape))})))
''' +check[end:]
start=check.index('with torch.no_grad():\n g=wave.gamma.clone()')
end=check.index('del base,off',start)
check=check[:start]+check[end:]
check=check.replace("ck('all_added_tensors_updated',all(not torch.equal(before[n],p.detach()) for n,p in added_parameters(model)))", """ck('all_added_tensors_updated',all(not torch.equal(before[n],p.detach()) for n,p in added_parameters(model)))
ck('all_1024_static_scalars_learn',bool(torch.all(module.c.detach()!=before['decoder.enhance.c'])))
ck('branch_and_D3_shapes',seen['A']['input']==[1,512,8,8] and seen['A']['output']==[1,512,8,8] and seen['B']['output']==[1,512,8,8] and seen['P']['input']==[1,1024,8,8] and seen['post']['output']==[1,512,8,8] and seen['D3']['input']==[1,512,8,8])
for hook in hooks:hook.remove()
# Explicit equation equivalence on rectangular-independent fixed 8x8 interface, no extra residual.
module.eval();probe=torch.rand(2,512,8,8)
with torch.no_grad():
 expected=module.post(probe+module.projection(module.c*torch.cat([module.a(probe),module.b(probe)],1)))
 ck('forward_equation_exact',torch.equal(module(probe),expected))""")
check=check.replace("checks=checks,loss=", "checks=checks,interfaces=seen,static_c_numel=module.c.numel(),loss=")
put('check.py',check)
controller=(old/'arm_controller.py').read_text().replace('source.wpo_skips_20261105','source.wpo_mr_20261105')
a=controller.index("   if phase=='preflight':")
b=controller.index("   if phase=='cpu':args=",a)
controller=controller[:a]+'''   if phase!='cpu':
    import shutil
    assert shutil.disk_usage(ROOT).free>=8*1024**3,'root free disk below8GiB; blocked'
''' +controller[b:]
put('arm_controller.py',controller)
print(json.dumps(dict(source=str(S),operations=str(O),run=str(R),created=sorted(p.name for p in S.iterdir())),indent=2))

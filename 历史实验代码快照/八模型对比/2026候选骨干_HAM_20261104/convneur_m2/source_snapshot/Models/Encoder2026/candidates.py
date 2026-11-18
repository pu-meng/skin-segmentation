"""Official candidate encoders, with explicit geometry adapters and frozen decoder."""
import importlib.util
import json
from pathlib import Path
import sys
import types

import torch
from torch import nn
import torch.nn.functional as F

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'source/comparison'))
sys.path.insert(0,'/home/PuMengYu/.cache/skin_encoder2026/deps')
from pretrained_unet2d import SharedDecoder, sha256_file

MATERIALS=PROJECT/'references/核对记录/2026候选_20261104'
SPECS={
    'convneur_m2': {'repo':'ConvNeur','channels':[48,96,192,384],
                   'geometry':'native four stages at strides 4,8,16,32'},
    'microvitv2_s1': {'repo':'MicroViT','channels':[32,64,128,320],
                     'geometry':'stem strides 4,8; main stage1 stride16; final stage3 stride64 bilinear to stride32'},
    'locatvit_tiny': {'repo':'LocAtViT','channels':[192]*4,
                     'geometry':'blocks 3,6,9,12; final norm and PRR retained; stride16 maps resized to 4,8,16,32'},
    'visionhope_tiny': {'repo':'VisionHOPE','channels':[64,128,256,512],
                       'geometry':'native four stages at strides 4,8,16,32'},
}


class RMSNormCompat(nn.Module):
    """PyTorch RMSNorm's FP32 formula/state layout for the existing torch 2.3."""
    def __init__(self,normalized_shape,eps=None,elementwise_affine=True,device=None,dtype=None):
        super().__init__()
        self.normalized_shape=(normalized_shape,) if isinstance(normalized_shape,int) else tuple(normalized_shape)
        self.eps=eps
        if elementwise_affine:
            self.weight=nn.Parameter(torch.ones(self.normalized_shape,device=device,dtype=dtype))
        else:
            self.register_parameter('weight',None)

    def forward(self,x):
        axes=tuple(range(x.ndim-len(self.normalized_shape),x.ndim))
        eps=torch.finfo(x.dtype).eps if self.eps is None else self.eps
        out=x*torch.rsqrt(x.square().mean(dim=axes,keepdim=True)+eps)
        return out if self.weight is None else out*self.weight


def package(name,root):
    if name in sys.modules:
        loaded=sys.modules[name]
        if str(root) not in list(getattr(loaded,'__path__',[])):
            raise RuntimeError('Conflicting official package: '+name)
        return loaded
    mod=types.ModuleType(name)
    mod.__path__=[str(root)]
    mod.__package__=name
    sys.modules[name]=mod
    return mod


def file_module(name,path):
    if name in sys.modules:return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod


def classifier(name,pretrained=True):
    spec=SPECS[name]
    root=PROJECT/'references/code'/spec['repo']
    if name=='convneur_m2':
        if not hasattr(nn,'RMSNorm'):nn.RMSNorm=RMSNormCompat
        package('model',root/'model')
        mod=file_module('model.convneur',root/'model/convneur.py')
        model=mod.ConvNeur(depths=[2,2,6,2],dims=[48,96,192,384],mem_dim=96,chunk_size=196)
        # Upstream repeat(..., h=1) aliases distinct template/learned Parameters.
        # EMA checkpoints contain different values for them. Separate the learned
        # storage so both branches load exactly; functional_call uses learned ones.
        for stage in model.stages:
            mem=stage[0].mem
            mem.memory_model_parameters=nn.ParameterList([nn.Parameter(p.detach().clone()) for p in mem.memory_model_parameters])
    elif name=='microvitv2_s1':
        mod=file_module('official_microvitv2',root/'model/microvitv2.py')
        # Released S1 checkpoint has 2/5/5 blocks; current factory says 2/6/5.
        # Preserve checkpoint topology rather than add an unpretrained block.
        model=mod.MicroViTv2(dims=[128,224,320],depths=[2,5,5],type=['c','c','sdta'],
                            attn_cr=[0,0,.25],patch_size=16,act_layer=nn.SiLU,final_feature=None)
    elif name=='locatvit_tiny':
        package('models',root/'models')
        mod=file_module('models.vit',root/'models/vit.py')
        if not getattr(mod.GaussianAugmentedBlock,'accepts_factory_device',False):
            original=mod.GaussianAugmentedBlock
            class FactoryCompatibleBlock(original):
                accepts_factory_device=True
                def __init__(self,*args,device=None,dtype=None,**kwargs):
                    if device is not None or dtype is not None:
                        raise ValueError('Only CPU/default-dtype official construction is supported')
                    super().__init__(*args,**kwargs)
            mod.GaussianAugmentedBlock=FactoryCompatibleBlock
        model=mod.locatvit_tiny(pretrained=False,dynamic_img_size=True,store_metrics=False)
    elif name=='visionhope_tiny':
        if 'visionhope' not in sys.modules:
            spec=importlib.util.spec_from_file_location('visionhope',root/'__init__.py',submodule_search_locations=[str(root)])
            mod=importlib.util.module_from_spec(spec);sys.modules['visionhope']=mod;spec.loader.exec_module(mod)
        from visionhope.models import create_model
        model=create_model(name)
    audit={'name':name,'official_classifier_parameters':sum(p.numel() for p in model.parameters()),
           'geometry':SPECS[name]['geometry'],'pretrained':pretrained,
           'rmsnorm_compatibility':name=='convneur_m2' and nn.RMSNorm is RMSNormCompat}
    if not pretrained:return model,audit
    if name=='visionhope_tiny':
        record=json.loads((PROJECT/'references/核对记录/2026_encoder/visionhope_tiny.json').read_text())
        weight=Path(record['weights'])
        expected=record['sha256']
        from visionhope.tools.checkpoint import read_weights
        state=read_weights(str(weight))
    else:
        record=json.loads((MATERIALS/(name+'_material.json')).read_text())
        weight=Path(record['path']);expected=record['sha256']
        if name=='locatvit_tiny':
            from safetensors.torch import load_file
            state=load_file(str(weight))
        else:
            state=torch.load(weight,map_location='cpu',weights_only=name!='microvitv2_s1')
        if isinstance(state,dict):
            # Follow upstream MicroViT's explicit evaluation branch ('model').
            for key in ['model','state_dict','model_ema']:
                if key in state and isinstance(state[key],dict):
                    state=state[key];audit['checkpoint_branch']=key;break
    if sha256_file(weight)!=expected:raise ValueError('Weight SHA256 mismatch')
    model.load_state_dict(state,strict=True)
    assert all(torch.equal(v,state[k]) for k,v in model.state_dict().items())
    audit.update(full_classifier_strict_load=True,all_loaded_tensors_equal=True,
                 weight=str(weight),weight_sha256=expected)
    return model,audit


class Encoder(nn.Module):
    def __init__(self,model,name,audit):
        super().__init__();self.backbone=model;self.name=name;self.loading_audit=audit
        self.native_shapes=None
        self.backbone.head=nn.Identity()
        if name=='convneur_m2':self.backbone.norm=nn.Identity()
        if name=='visionhope_tiny':
            for key in ('norm','head_pre_logits','head_norm','head_act'):setattr(self.backbone,key,nn.Identity())

    def forward(self,x):
        m=self.backbone
        if self.name=='convneur_m2':features=m.forward_features(x)
        elif self.name=='microvitv2_s1':
            features=[]
            for i,layer in enumerate(m.stages[0].stem):
                x=layer(x)
                if i in (1,2):features.append(x)
            for i,layer in enumerate(list(m.stages)[1:],start=1):
                x=layer(x)
                if i==1:features.append(x)
            features.append(x)
        elif self.name=='locatvit_tiny':
            x=m.patch_embed(x);x=m._pos_embed(x);x=m.patch_drop(x);x=m.norm_pre(x)
            features=[]
            for i,blk in enumerate(m.blocks):
                x=blk(x)
                if i in (2,5,8,11):
                    y=m.prr(m.norm(x)) if i==11 else x
                    y=y[:,m.num_prefix_tokens:,:]
                    side=round(y.shape[1]**.5)
                    features.append(y.transpose(1,2).reshape(y.shape[0],y.shape[2],side,side))
        else:features=m.forward_intermediates(x)
        self.native_shapes=[list(f.shape) for f in features]
        output=[]
        for i,f in enumerate(features):
            size=256//(4*2**i)
            if f.shape[-2:]!=(size,size):
                f=F.interpolate(f,size=(size,size),mode='bilinear',align_corners=False)
            output.append(f)
        return output


class CandidateUNet(nn.Module):
    def __init__(self,name,pretrained=True):
        super().__init__()
        model,audit=classifier(name,pretrained)
        self.encoder=Encoder(model,name,audit)
        self.register_buffer('mean',torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('std',torch.tensor([.229,.224,.225]).view(1,3,1,1))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42);self.decoder=SharedDecoder()
            torch.manual_seed(43);self.align=nn.ModuleList([nn.Conv2d(a,b,1) for a,b in zip(SPECS[name]['channels'],[64,128,256,512])])

    def forward(self,x):
        return self.decoder([layer(f) for layer,f in zip(self.align,self.encoder((x-self.mean)/self.std))])


def build_model(name,pretrained=True):return CandidateUNet(name,pretrained)

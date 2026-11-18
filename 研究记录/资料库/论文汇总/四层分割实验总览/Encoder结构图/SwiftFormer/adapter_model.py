"""Official TinyViM comparison encoders with the unchanged shared UNet decoder."""
import hashlib
import importlib.util
import importlib
import json
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'source/comparison'))
from pretrained_unet2d import SharedDecoder,prepare_imports
WEIGHTS=ROOT/'Models/pretrained/tinyvim_peers_20261101'
MANIFEST=ROOT/'references/核对记录/tinyvim_peers_20261101/materials.json'
SPECS={
    'swiftformer_s':('SwiftFormer','models/swiftformer.py','SwiftFormer_S','swiftformer_s.pth',(48,64,168,224)),
    'efficientformerv2_s0':('EfficientFormer','models/efficientformer_v2.py','efficientformerv2_s0','efficientformerv2_s0.pth',(32,48,96,176)),
    'mobileone_s1':('MobileOne','mobileone.py','mobileone','mobileone_s1_unfused.pth.tar',(96,192,512,1280)),
    'efficientvmamba_t':('EfficientVMamba','classification/models/vmamba_efficient.py','EfficientVSSM','efficient_vmamba_tiny.ckpt',(48,96,192,384)),
}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def official_module(name):
    repo,source,_,_,_=SPECS[name]
    path=ROOT/'references/code'/repo/source
    key='official_tinyvim_peer_'+name
    if key not in sys.modules:
        # Older official sources use the previous timm location of this same helper.
        try:
            importlib.import_module('timm.models.layers.helpers')
        except ModuleNotFoundError:
            sys.modules['timm.models.layers.helpers']=importlib.import_module('timm.layers.helpers')
        if name=='efficientvmamba_t':
            sys.path.insert(0,str(ROOT/'references/code/EfficientVMamba/classification'))
        spec=importlib.util.spec_from_file_location(key,path)
        module=importlib.util.module_from_spec(spec)
        sys.modules[key]=module
        spec.loader.exec_module(module)
    return sys.modules[key]


def factory(name,resolution=224):
    module=official_module(name)
    make=getattr(module,SPECS[name][2])
    if name=='mobileone_s1':return make(variant='s1',inference_mode=False)
    if name=='efficientvmamba_t':
        return make(depths=[2,2,4,2],dims=48,d_state=16,dt_rank='auto',ssm_ratio=2.,
                    mlp_ratio=0.,drop_path_rate=.2,downsample_version='v1',step_size=2)
    if name=='efficientformerv2_s0':return make(pretrained=False,resolution=resolution)
    return make(pretrained=False)


def load_classifier(name,pretrained=True):
    model=factory(name)
    audit={'pretrained':pretrained,'geometry_adaptation':[]}
    if pretrained:
        record=json.loads((MANIFEST.parent/(name+'.json')).read_text())
        path=WEIGHTS/SPECS[name][3]
        assert path.stat().st_size==record['bytes'] and sha(path)==record['sha256']
        payload=torch.load(path,map_location='cpu',weights_only=False)
        if name=='efficientvmamba_t':
            state=payload['state_dict'] if 'state_dict' in payload else payload.get('model',payload)
            state={k.removeprefix('module.module.').removeprefix('module.'):v for k,v in state.items()}
        else:
            state=payload.get('model',payload.get('state_dict',payload))
        model.load_state_dict(state,strict=True)
        assert all(torch.equal(v,state[k]) for k,v in model.state_dict().items())
        audit.update(record,full_classifier_strict_load=True,all_loaded_tensors_equal=True,
                     classification_parameters=sum(p.numel() for p in model.parameters()))
    if name=='efficientformerv2_s0':
        target=factory(name,resolution=256)
        old=model.state_dict();adapted={}
        for key,value in target.state_dict().items():
            if key.endswith('attention_bias_idxs'):
                adapted[key]=value
            elif old[key].shape==value.shape:
                adapted[key]=old[key]
            elif key.endswith('attention_biases'):
                a,b=int(old[key].shape[1]**.5),int(value.shape[1]**.5)
                assert a*a==old[key].shape[1] and b*b==value.shape[1]
                adapted[key]=F.interpolate(old[key].reshape(1,-1,a,a),size=(b,b),mode='bicubic',align_corners=False).reshape_as(value)
                audit['geometry_adaptation'].append(dict(key=key,old=list(old[key].shape),new=list(value.shape),method='bicubic relative-offset grid interpolation'))
            else:
                raise ValueError('Unexpected geometry change: '+key)
        target.load_state_dict(adapted,strict=True)
        assert all(torch.equal(v,adapted[k]) for k,v in target.state_dict().items())
        model=target
    return model,audit


class Encoder(nn.Module):
    def __init__(self,name,backbone):
        super().__init__()
        self.name=name;self.backbone=backbone
        if name in ('swiftformer_s','efficientformerv2_s0'):
            self.backbone.norm=nn.Identity()
            self.backbone.head=nn.Identity()
            if hasattr(self.backbone,'dist_head'):self.backbone.dist_head=nn.Identity()
        elif name=='mobileone_s1':
            self.backbone.linear=nn.Identity()
        else:
            self.backbone.classifier=nn.Identity()
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(43)
                self.outnorm=nn.ModuleList(nn.LayerNorm(c) for c in SPECS[name][4])

    def forward(self,x):
        b=self.backbone;outs=[]
        if self.name=='mobileone_s1':
            x=b.stage0(x)
            for stage in (b.stage1,b.stage2,b.stage3,b.stage4):
                x=stage(x);outs.append(x)
        elif self.name in ('swiftformer_s','efficientformerv2_s0'):
            x=b.patch_embed(x)
            for i,stage in enumerate(b.network):
                x=stage(x)
                if i in (0,2,4,6):outs.append(x)
        else:
            # Same pre-downsample feature extraction as official Backbone_EfficientVSSM.
            x=b.patch_embed(x)
            for i,stage in enumerate(b.layers):
                if hasattr(stage,'blocks'):
                    o=stage.blocks(x);x=stage.downsample(o)
                else:
                    o=stage[:-1](x);x=stage[-1](o)
                outs.append(self.outnorm[i](o).permute(0,3,1,2).contiguous())
        assert [tuple(v.shape[1:]) for v in outs]==[(c,s,s) for c,s in zip(SPECS[self.name][4],(64,32,16,8))]
        return outs


class PeerUNet(nn.Module):
    def __init__(self,name,backbone,audit):
        super().__init__()
        self.encoder=Encoder(name,backbone)
        self.encoder.loading_audit=audit
        self.register_buffer('mean',torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('std',torch.tensor([.229,.224,.225]).view(1,3,1,1))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42);self.decoder=SharedDecoder()
            torch.manual_seed(43);self.align=nn.ModuleList(nn.Conv2d(a,b,1) for a,b in zip(SPECS[name][4],(64,128,256,512)))

    def forward(self,x):
        return self.decoder([align(v) for align,v in zip(self.align,self.encoder((x-self.mean)/self.std))])


def build_model(name,pretrained=True):
    if name not in SPECS:raise ValueError(name)
    prepare_imports('mambavision_t')
    model,audit=load_classifier(name,pretrained)
    return PeerUNet(name,model,audit)

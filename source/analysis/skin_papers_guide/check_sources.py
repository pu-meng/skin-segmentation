"""Small CPU checks of diagram arithmetic; not a training or performance reproduction."""
import ast
import hashlib
import json
import math
import random
import subprocess
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[3]
SV=ROOT/'references/code/SvANet'
OUT=ROOT/'log/研究记录_辅助产物/skin_papers_guide'
torch.set_num_threads(2)
torch.manual_seed(7); np.random.seed(7); random.seed(7)

def extract(path,names,env):
    tree=ast.parse(path.read_text())
    body=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    assert len(body)==len(names), (path,names)
    exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),env)

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    files=[ROOT/'references/code/MambaLiteUNet/models/MambaLiteUNet.py']
    files += [SV/p for p in ['opts.py','shell/main.sh','lib/model/segmentation/heads/baseSegHead.py',
      'lib/model/segmentation/heads/module/fgModules.py','lib/model/modules/mcattn.py','lib/model/modules/assemFormer.py']]
    env=dict(globals())
    env.update(initWeight=lambda m:None,pair=lambda v:(v,v),StochasticDepth=lambda p:nn.Identity(),
      Dropout=nn.Dropout,AdaptiveAvgPool2d=nn.AdaptiveAvgPool2d,AdaptiveMaxPool2d=nn.AdaptiveMaxPool2d,
      setMethod=setattr,callMethod=getattr)
    extract(SV/'lib/utils/mathematics/utils.py',['makeDivisible'],env)
    extract(SV/'lib/utils/tensor_operation/utils.py',['shuffleTensor'],env)
    extract(SV/'lib/model/layers/convLayer.py',['BaseConv2d'],env)
    extract(SV/'lib/model/modules/mcattn.py',['MoCAttention'],env)
    extract(SV/'lib/model/modules/assemFormer.py',['LinearSelfAttention','LinearAttnFFN','BaseFormer','AssembleFormer'],env)
    extract(SV/'lib/model/segmentation/heads/module/fgModules.py',['BasicBlock','CSLayer'],env)
    checks={}
    with torch.no_grad():
        linear=env['LinearSelfAttention'](8).eval(); x=torch.randn(2,8,4,16)
        q,k,v=torch.split(linear.qkv_proj(x),[1,8,8],dim=1)
        a=q.softmax(-1);z=(k*a).sum(-1,keepdim=True)
        ref=linear.out_proj(v.relu()*z)
        checks['linear_attention_max_abs_error']=(linear(x)-ref).abs().max().item()
        assert torch.allclose(linear(x),ref)
        checks['linear_attention_weight_sum_max_error']=(a.sum(-1)-1).abs().max().item()
        former=env['AssembleFormer'](16).eval(); x=torch.randn(2,16,8,8)
        local=former.LocalRep(x); patches,size=former.unfolding(local)
        assert patches.shape==(2,8,4,16)
        err=(former.folding(patches,size)-local).abs().max().item()
        checks['patch_unfold_fold_max_error']=err
        checks['assemformer_output_shape']=list(former(x).shape)
        assert err==0 and former(x).shape==x.shape
        mca=env['MoCAttention'](16).eval()
        ref=x*mca.SELayer(x.mean((-2,-1),keepdim=True))
        checks['mcattn_eval_gap_max_error']=(mca(x)-ref).abs().max().item()
        assert torch.allclose(mca(x),ref,atol=1e-6)
        mca.train();checks['mcattn_training_sample_shape']=list(mca.monteCarloSample(x).shape)
        opt=SimpleNamespace(fg_svattn=1,reparamed=False,resize_shape=64,fg_svattn_divisor=4,device='cpu')
        cs=env['CSLayer'](opt,16,32,Stride=2,NumBlocks=2,Stage=1).eval()
        x=torch.randn(2,16,32,32)
        features=[x,cs.Conv1(x)];features.append(cs.Conv2(features[-1]))
        p=[getattr(cs,'Squeeze%d'%(i+1))(cs.PoolLayer(f)) for i,f in enumerate(features)]
        attn=torch.sigmoid(cs.PoolOne(sum(p)/len(p)))
        ref=features[-1]*attn
        checks['svattn_eval_stage_mean_max_error']=(cs(x)-ref).abs().max().item()
        assert torch.allclose(cs(x),ref)
        checks['svattn_output_shape']=list(cs(x).shape)
        masks=sum(cs.AttnMask)
        checks['svattn_mask_partition_max_error']=(masks-1).abs().max().item()
        assert torch.equal(masks,torch.ones_like(masks))
        cs.train();assert cs(x).shape==(2,32,8,8)

        # Only Mamba is replaced to verify the actual surrounding forward's routing.
        class IdentityMamba(nn.Module):
            def __init__(self,**kwargs):super().__init__()
            def forward(self,x):return x
        env.update(Mamba=IdentityMamba,trunc_normal_=nn.init.trunc_normal_)
        names=['DepthwiseConvolution','MambaSSMLayer','LocalGlobalFeatureMixing',
          'AdaptiveMultiBranchMambaFeatureFusion','CrossGatedAttentionGate','MambaLiteUNet']
        extract(files[0],names,env)
        amf=env['AdaptiveMultiBranchMambaFeatureFusion'](16,24).eval()
        x=torch.randn(2,16,4,4)
        checks['amf_skip_scale_init']=amf.skip_scale.item()
        checks['amf_shape_with_identity_mamba']=list(amf(x).shape)
        assert amf(x).shape==(2,24,4,4)
        checks['q_conv_groups_and_output']=[amf.conv_q_dw.groups,amf.conv_q_dw.out_channels]
        cga=env['CrossGatedAttentionGate'](16,16,8).eval()
        out=cga(x,x)
        assert out.shape==x.shape
        checks['cga_shape_with_identity_mamba']=list(out.shape)
        full=env['MambaLiteUNet']().eval()
        shapes={}
        hooks=[]
        for name in ['encoder1','encoder2','encoder3','encoder4','encoder5','encoder6','decoder1','decoder2','decoder3','decoder4','decoder5']:
            hooks.append(getattr(full,name).register_forward_hook(lambda m,a,o,n=name:shapes.update({n:list(o.shape)})))
        out=full(torch.randn(2,3,32,32))
        for h in hooks:h.remove()
        assert out.shape==(2,1,32,32)
        checks['mambalite_routing_with_identity_mamba']={'module_shapes_at_32_input':shapes,'output':list(out.shape)}
    result={'scope':'CPU module arithmetic and routing checks only. No model training or metric reproduction.',
      'substitutions':{'SvANet':'Classes extracted from source AST; initWeight is a no-op; DropPath is identity in eval checks.',
                       'MambaLiteUNet':'Mamba alone replaced by identity for routing checks; this does NOT validate SSM arithmetic or accuracy.'},
      'svanet_commit':subprocess.check_output(['git','-C',str(SV),'rev-parse','HEAD'],text=True).strip(),
      'sources_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'checks':checks}
    (OUT/'source_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(checks,ensure_ascii=False,indent=2))

if __name__=='__main__':run()

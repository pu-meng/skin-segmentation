"""CPU audit: unchanged official class, sharing, gradients and inherited initialization."""
from pathlib import Path
import ast
import io
import json
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison')]
import torch
from Models.SharedADGBC.model import TinyViMSharedADGBC
from Models.SharedADGBC.official_granular_ball import GranularBall
from pretrained_unet2d import PretrainedUNet2D


def check():
    torch.set_num_threads(2);torch.manual_seed(42)
    def class_ast(path):
        tree=ast.parse(path.read_text())
        node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GranularBall')
        return ast.dump(node,include_attributes=False)
    assert class_ast(ROOT/'Models/SharedADGBC/official_granular_ball.py')==class_ast(ROOT/'references/code/AD-GBC/archs_GBC.py')
    block=GranularBall(64,num_balls=32,proj_dim=64).eval()
    encoder,decoder=torch.randn(2,64,8,8),torch.randn(2,64,8,8)
    e,a,sigma,dif=block(encoder)
    d,*_=block(decoder)
    assert e.shape==encoder.shape and d.shape==decoder.shape
    assert a.shape==(2,64,32) and sigma.shape==(1,1,32,64)
    torch.testing.assert_close(a.sum(-1),torch.ones(2,64))
    assert (sigma>0).all()
    parameters=list(block.parameters())
    ge=torch.autograd.grad(e.square().mean(),parameters,retain_graph=True)
    gd=torch.autograd.grad(d.square().mean(),parameters,retain_graph=True)
    gt=torch.autograd.grad(e.square().mean()+d.square().mean(),parameters)
    for x,y,z in zip(ge,gd,gt):
        assert torch.isfinite(x).all() and torch.isfinite(y).all()
        assert x.abs().sum()>0 and y.abs().sum()>0
        torch.testing.assert_close(x+y,z,rtol=1e-4,atol=1e-6)
    stream=io.BytesIO();torch.save(block.state_dict(),stream);stream.seek(0)
    reloaded=GranularBall(64,num_balls=32,proj_dim=64).eval()
    reloaded.load_state_dict(torch.load(stream,weights_only=True),strict=True)
    torch.testing.assert_close(reloaded(encoder)[0],block(encoder)[0],rtol=0,atol=0)
    block.train();before=block.refine[1].num_batches_tracked.clone()
    block(encoder);block(decoder)
    assert int(block.refine[1].num_batches_tracked-before)==2
    model=TinyViMSharedADGBC(pretrained=False)
    baseline=PretrainedUNet2D('tinyvim_s',pretrained=False,new_layers_seed=42)
    for name in ('decoder','align'):
        ma,ba=getattr(model,name).state_dict(),getattr(baseline,name).state_dict()
        assert ma.keys()==ba.keys() and all(torch.equal(ma[k],ba[k]) for k in ma)
    assert sum(p.numel() for p in model.shared_adgbc.parameters())==41088
    assert sum(p.numel() for p in model.parameters())==8716429
    assert sum(isinstance(m,GranularBall) for m in model.modules())==1
    assert isinstance(model.align[-1],torch.nn.Conv2d)
    return {'status':'passed','official_GranularBall_AST_identical':True,
        'one_shared_module_instance':True,'encoder_and_decoder_gradients_sum_into_shared_parameters':True,
        'positive_anisotropic_scales':True,'membership_normalized_over32_balls':True,
        'shared_BN_updated_twice':True,'strict_reload':True,
        'paired_baseline_decoder_alignment_initialization':True,
        'added_parameters':41088,'parameters':8716429,'test_data_accessed':False}


if __name__=='__main__':
    print(json.dumps(check(),ensure_ascii=False,indent=2))

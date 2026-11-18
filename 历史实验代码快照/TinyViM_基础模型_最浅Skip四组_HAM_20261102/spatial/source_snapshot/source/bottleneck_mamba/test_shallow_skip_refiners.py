"""Meaningful CPU contracts for axes, context, spatial support and inherited initialization."""
import io
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT),str(ROOT/'source/comparison')]
import torch
from Models.ShallowSkipRefiners.model import REFINERS, TinyViMShallowSkipRefiner
from pretrained_unet2d import PretrainedUNet2D


def check():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    skip, context = torch.randn(2,64,64,64), torch.randn(2,64,64,64)
    results = {}
    baseline = PretrainedUNet2D('tinyvim_s',pretrained=False,new_layers_seed=42)
    for kind,cls in REFINERS.items():
        module = cls()
        optimizer = torch.optim.AdamW(module.parameters(),lr=.001)
        before = [v.detach().clone() for v in module.parameters()]
        output = module(skip,context)
        assert output.shape == skip.shape and torch.isfinite(output).all()
        output.square().mean().backward()
        assert all(v.grad is not None and torch.isfinite(v.grad).all() and v.grad.abs().sum()>0 for v in module.parameters())
        optimizer.step()
        assert all(not torch.equal(a,b) for a,b in zip(before,module.parameters()))
        if kind == 'spatial':
            weights = module.weights(skip,context)
            assert weights.shape == (2,1,64,64) and (weights>0).all() and (weights<1).all()
            assert not torch.equal(weights,module.weights(skip,torch.zeros_like(context)))
        if kind == 'channel':
            weights = module.weights(skip)
            assert weights.shape == (2,64,1,1) and (weights>0).all() and (weights<1).all()
        if kind == 'conv':
            zero = torch.zeros_like(skip)
            impulse = zero.clone();impulse[:,:,30,30]=1
            delta = module(impulse)-module(zero)
            outside = delta.clone();outside[:,:,29:32,29:32]=0
            assert outside.count_nonzero()==0
        if kind == 'region':
            tokens,a,regions = module.aggregate(skip)
            torch.testing.assert_close(a.sum(-1),torch.ones(2,4096))
            weights=a/a.sum(1,keepdim=True).clamp_min(1e-6)
            torch.testing.assert_close(weights.sum(1),torch.ones(2,8))
            torch.testing.assert_close(regions,weights.transpose(1,2)@tokens)
            torch.testing.assert_close(module(skip),(a@regions).transpose(1,2).reshape_as(skip))
            assert regions.shape == (2,8,64)
        stream=io.BytesIO();torch.save(module.state_dict(),stream);stream.seek(0)
        other=cls();other.load_state_dict(torch.load(stream,weights_only=True),strict=True)
        torch.testing.assert_close(other(skip,context),module(skip,context),rtol=0,atol=0)
        model=TinyViMShallowSkipRefiner(kind,pretrained=False)
        for name in ('decoder','align'):
            a,b=getattr(model,name).state_dict(),getattr(baseline,name).state_dict()
            assert a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)
        assert isinstance(model.align[-1],torch.nn.Conv2d)
        results[kind]={'status':'passed','adapter_parameters':sum(v.numel() for v in module.parameters()),
                       'parameters':sum(v.numel() for v in model.parameters()),
                       'all_parameter_tensors_updated':True,'strict_reload':True,
                       'paired_baseline_decoder_alignment_initialization':True}
        del model
    return {'status':'passed','arms':results,'test_data_accessed':False}


if __name__=='__main__':
    print(json.dumps(check(),ensure_ascii=False,indent=2))

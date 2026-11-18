"""CPU contracts for spatial/channel alignment, summation and inherited initialization."""
import io
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'source/comparison')]
import torch
from Models.SkipFusion.model import MultiScaleSkipFusion, TinyViMV4SkipFusion
from pretrained_unet2d import PretrainedUNet2D


def checks():
    torch.set_num_threads(2)
    module = MultiScaleSkipFusion()
    inputs = [torch.randn(2,c,h,h, requires_grad=True) for c,h in ((64,64),(128,32),(256,16))]
    output = module(inputs)
    expected = sum(layer(x) for layer,x in zip(module.branches, inputs))
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert output.shape == (2,256,16,16)
    count = sum(p.numel() for p in module.parameters())
    assert count == 118720
    output.square().mean().backward()
    assert all(x.grad is not None and x.grad.abs().sum()>0 for x in inputs)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in module.parameters())
    stream = io.BytesIO(); torch.save(module.state_dict(), stream); stream.seek(0)
    other = MultiScaleSkipFusion(); other.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    torch.testing.assert_close(other(inputs), output, rtol=0, atol=0)
    model = TinyViMV4SkipFusion(pretrained=False)
    baseline = PretrainedUNet2D('tinyvim_s', pretrained=False, new_layers_seed=42)
    # The fresh official encoder is overwritten from official weights in formal runs.
    # Pair-check the seed-controlled decoder and all four alignment projections.
    for name in ('decoder', 'align'):
        a, b = getattr(model,name).state_dict(), getattr(baseline,name).state_dict()
        assert a.keys() == b.keys()
        assert all(torch.equal(a[k],b[k]) for k in a)
    assert sum(p.numel() for p in model.parameters()) == 8794061
    assert isinstance(model.align[-1], torch.nn.Conv2d)
    assert not any(type(layer).__name__.startswith('BottleneckMamba') for layer in model.modules())
    return {'status':'passed', 'fusion_parameters':count, 'model_parameters':8794061,
            'three_input_shapes_verified':True, 'elementwise_sum_verified':True,
            'all_branch_gradients_finite_nonzero':True, 'strict_save_reload':True,
            'baseline_decoder_alignment_initial_tensors_equal':True,
            'V2_not_installed':True, 'test_data_accessed':False}


if __name__ == '__main__':
    import json
    print(json.dumps(checks(), ensure_ascii=False, indent=2))

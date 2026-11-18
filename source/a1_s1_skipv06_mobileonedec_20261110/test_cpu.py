"""CPU acceptance checks for A1-S1-SKIPv0.6-MobileOneDec.

Run:
  /home/PuMengYu/anaconda3/envs/medseg/bin/python \
    source/a1_s1_skipv06_mobileonedec_20261110/test_cpu.py

Covers: official MobileOne block loading, single-stage forward/backward at 32x32
and 64x64, skip-branch presence when in==out, branch re-parameterization equality
in eval mode, the real D1 -> ups[0] -> Skip1V06 -> D0 chain, parent parameter
preservation, and parameter counts. Full-network forward still needs CUDA (Mamba).
"""
import copy
import sys

BASE = '/home/PuMengYu/experiments/skin_segmentation'
sys.path[:0] = [BASE, BASE + '/source/comparison', BASE + '/source/bottleneck_mamba']

from pretrained_unet2d import prepare_imports

prepare_imports('mambavision_t')
prepare_imports('tinyvim_s')

import torch

from source.a1_s1_skipv06_mobileonedec_20261110.candidate import (
    STAGE_SHAPES, MobileOneDecoderStage, build_candidate, official_mobileone,
    reparameterize, replaced_levels)
from source.a1_s1_skip1_v06_20261108.candidate import build_candidate as build_parent

torch.set_num_threads(4)
torch.manual_seed(42)
device = 'cpu'

official = official_mobileone()
assert hasattr(official, 'MobileOneBlock') and hasattr(official, 'reparameterize_model')
print('official_mobileone_loaded', official.MobileOneBlock.__module__)

# 1. Single stage at both shallow resolutions, forward and backward.
for c_in, c_out, size in [(256, 128, 32), (128, 64, 64)]:
    stage = MobileOneDecoderStage(c_in, c_out, depth=1).to(device).train()
    x = torch.randn(2, c_in, size, size, device=device, requires_grad=True)
    y = stage(x)
    assert y.shape == (2, c_out, size, size), (y.shape, (2, c_out, size, size))
    y.square().mean().backward()
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
    assert all(p.grad is not None for p in stage.parameters()), 'some parameters got no gradient'
    assert all(torch.isfinite(p.grad).all() for p in stage.parameters() if p.grad is not None)
    stage.eval()
    print(f'stage {c_in}->{c_out} @{size}x{size} forward_backward_pass '
          f'params={sum(p.numel() for p in stage.parameters())}')

# 2. Skip branch appears only when in==out; depth=2 exercises both cases.
two = MobileOneDecoderStage(256, 128, depth=2)
assert not hasattr(two.blocks[0], 'rbr_skip') or two.blocks[0].rbr_skip is None
assert two.blocks[1].rbr_skip is not None, 'second block (128->128) must carry the BN skip branch'
assert len(two.blocks) == 2
with torch.no_grad():
    assert two.eval()(torch.randn(1, 256, 16, 16)).shape == (1, 128, 16, 16)
print('depth_2_skip_branch_ok')

# 3. Re-parameterization must be numerically equivalent in eval mode.
for c_in, c_out, size in [(256, 128, 32), (128, 64, 64)]:
    stage = MobileOneDecoderStage(c_in, c_out, depth=2).eval()
    with torch.no_grad():
        for m in stage.modules():
            if hasattr(m, 'running_mean'):
                m.running_mean.normal_(); m.running_var.uniform_(0.5, 2.0)
                if m.weight is not None: m.weight.normal_(1.0, 0.1)
                if m.bias is not None: m.bias.normal_()
    x = torch.randn(2, c_in, size, size)
    with torch.no_grad():
        before = stage(x)
        fused = copy.deepcopy(stage)
        count = reparameterize(fused)
        after = fused(x)
    assert count == 2, count
    err = (before - after).abs().max().item()
    assert err < 1e-3, err
    print(f'reparameterize_equal {c_in}->{c_out} blocks={count} max_abs_err={err:.2e}')

# 4. Real D1 -> ups[0] -> Skip1V06 -> D0 chain (no Mamba involved).
model = build_candidate(pretrained=False).to(device).train()
h = model.decoder.stages[1](torch.randn(2, 256, 32, 32, device=device))
h = model.decoder.ups[0](h)
s = torch.randn(2, 64, 64, 64, device=device)
s2, h2 = model.decoder.e3(s, h)
out = model.decoder.stages[0](torch.cat((h2, s2), dim=1))
assert h.shape == (2, 64, 64, 64) and out.shape == (2, 64, 64, 64), (h.shape, out.shape)
out.square().mean().backward()
assert torch.isfinite(out).all()
print('shallow_chain D1->ups0->skip1v06->D0 forward_backward_pass')

# 5. Parent parameter preservation outside the replaced stages.
torch.manual_seed(42)
base = build_parent(pretrained=False)
torch.manual_seed(42)
target = build_candidate(pretrained=False)
assert replaced_levels(target) == [1, 0], replaced_levels(target)
a, b = base.state_dict(), target.state_dict()
swapped = tuple(f'decoder.stages.{level}.' for level in (1, 0))
kept = [k for k in a if not k.startswith(swapped)]
assert all(k in b and torch.equal(a[k], b[k]) for k in kept), 'parent weights drifted'
assert all(k.startswith(swapped) or k in a for k in b), 'unexpected extra parameters'
print('parent_preservation_identical_tensors', len(kept))

# 6. Parameter accounting per replacement.
base_params = sum(p.numel() for p in base.parameters())
for config in [(1, 0), (0,), (2, 1, 0), (3, 2, 1, 0)]:
    torch.manual_seed(42)
    m = build_candidate(pretrained=False, stages=config)
    assert replaced_levels(m) == sorted(config, reverse=True)
    assert isinstance(m.decoder.e3, type(base.decoder.e3))
    params = sum(p.numel() for p in m.parameters())
    print(f'stages={config} params={params} delta_vs_parent={params - base_params:+d}')

# 7. Deeper stacks stay monotonic in parameters.
torch.manual_seed(42)
one = build_candidate(pretrained=False, stages=(1, 0), depth=1)
torch.manual_seed(42)
two_ = build_candidate(pretrained=False, stages=(1, 0), depth=2)
assert sum(p.numel() for p in two_.parameters()) > sum(p.numel() for p in one.parameters())
print('depth_monotonic_params_ok')

print('CPU checks passed; full-network forward/backward and real Mamba remain CUDA-only')

"""CPU acceptance checks for the lighter-upsampling / lighter-align variants.

Run:
  /home/PuMengYu/anaconda3/envs/medseg/bin/python \
    source/a1_s1_skipv06_lightups_20261110/test_cpu.py
"""
import sys

BASE = '/home/PuMengYu/experiments/skin_segmentation'
sys.path[:0] = [BASE, BASE + '/source/comparison', BASE + '/source/bottleneck_mamba']

from pretrained_unet2d import prepare_imports

prepare_imports('mambavision_t')
prepare_imports('tinyvim_s')

import torch

from source.a1_s1_skipv06_lightups_20261110.candidate import (
    ALIGN_SHAPES, UP_SHAPES, InterpUp, LowRankAlign, build_candidate)
from source.a1_s1_skip1_v06_20261108.candidate import build_candidate as build_parent

torch.set_num_threads(4)
torch.manual_seed(42)
device = 'cpu'


def decoder_chain(model):
    """Run D3-output -> ups2 -> D2 -> ups1 -> D1 -> ups0 -> skip -> D0. No Mamba involved."""
    x = torch.randn(2, 512, 8, 8, device=device)
    f2 = torch.randn(2, 256, 16, 16, device=device)
    f1 = torch.randn(2, 128, 32, 32, device=device)
    s = torch.randn(2, 64, 64, 64, device=device)
    d = model.decoder
    x = d.ups[2](x)
    x = d.stages[2](torch.cat((x, f2), dim=1))
    x = d.ups[1](x)
    x = d.stages[1](torch.cat((x, f1), dim=1))
    x = d.ups[0](x)
    s2, x2 = d.e3(s, x)
    x = d.stages[0](torch.cat((x2, s2), dim=1))
    return x


# 1. InterpUp keeps the exact interface of the ConvTranspose it replaces.
for level, (c_in, c_out) in UP_SHAPES.items():
    up = InterpUp(c_in, c_out).to(device)
    y = up(torch.randn(2, c_in, 8 * 2 ** (2 - level), 8 * 2 ** (2 - level), device=device))
    assert y.shape[1] == c_out and y.shape[-1] == 2 * (8 * 2 ** (2 - level)), (level, y.shape)
print('interp_up_shapes_ok')

# 2. The whole shallow decoder chain must keep channel flow for every switch combination.
totals = {}
for ups in ('conv', 'interp'):
    for align in ('full', 'lowrank', 'group8'):
        model = build_candidate(pretrained=False, ups=ups, align=align).to(device).train()
        out = decoder_chain(model)
        assert out.shape == (2, 64, 64, 64), (ups, align, out.shape)
        out.square().mean().backward()
        assert torch.isfinite(out).all()
        params = sum(p.numel() for p in model.parameters())
        totals[(ups, align)] = params
        mb = params * 4 / 1024 / 1024
        print(f'ups={ups:6s} align={align:7s} chain_ok params={params:>10,d} {mb:6.2f} MB')

# 3. Parent preservation: only decoder.ups.* and align.* may differ from the parent.
torch.manual_seed(42)
base = build_parent(pretrained=False)
a = base.state_dict()
for ups, align in [('interp', 'full'), ('conv', 'lowrank'), ('interp', 'group8')]:
    torch.manual_seed(42)
    model = build_candidate(pretrained=False, ups=ups, align=align)
    b = model.state_dict()
    changed = ('decoder.ups.', 'align.')
    kept = [k for k in a if not k.startswith(changed)]
    assert all(k in b and torch.equal(a[k], b[k]) for k in kept), (ups, align, 'parent drifted')
    assert all(k.startswith(changed) or k in a for k in b), (ups, align, 'extra params')
    print(f'parent_preservation_ok ups={ups} align={align} tensors={len(kept)}')

# 4. Report the deltas against the parent for every combination.
base_params = sum(p.numel() for p in base.parameters())
print(f'\nparent params={base_params:,}')
for key, value in sorted(totals.items()):
    print(f'{key[0]:6s}/{key[1]:7s} delta={value - base_params:>+9,d} ({(value-base_params)*4/1024/1024:+.2f} MB)')

print('\nCPU checks passed; full-network forward and real Mamba remain CUDA-only')

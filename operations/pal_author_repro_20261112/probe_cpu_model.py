#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

BASE = Path("/home/PuMengYu/experiments/skin_segmentation")
OP = BASE / "operations/pal_author_repro_20261112"
VENDOR = OP / "vendor/PAL"
DEPS = OP / "vendor_deps"
sys.path.insert(0, str(DEPS))
sys.path.insert(0, str(VENDOR))
sys.path.insert(0, str(VENDOR / "src"))
os.chdir(VENDOR / "src")

import torch
from our_model import Model

args = SimpleNamespace(
    backbone="pvt", patchsize=256, snapshot=None, mask_dim=256, num_queries=2,
    num_classes=2, cls_attributes=10, dec_layers=9, shareW=True, fuse_mode="conv",
    topk=3, sample_num=15, tau=1.0,
)
orig_cuda = torch.Tensor.cuda
orig_module_cuda = torch.nn.Module.cuda
torch.Tensor.cuda = lambda self, *a, **kw: self
torch.nn.Module.cuda = lambda self, *a, **kw: self
try:
    model = Model(args).eval()
    checkpoint = torch.load(
        OP.parent.parent / "references/核对记录/PAL_PVT官方预训练/pvt_v2_b2.pth",
        map_location="cpu", weights_only=True,
    )
    state = model.bkbone.state_dict()
    matched = [k for k, v in state.items() if k in checkpoint and v.shape == checkpoint[k].shape]
    assert len(matched) == len(state), (len(matched), len(state))
    assert all(torch.equal(state[k], checkpoint[k]) for k in matched)
    param_count = sum(p.numel() for p in model.parameters())
    print("PVT_WEIGHT_KEYS", len(matched), "OF", len(state))
    print("PAL_PARAMETERS", param_count)

    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out = model(x)
    y = out["fuse_prob"]
    assert y.shape == (1, 1, 256, 256), tuple(y.shape)
    assert torch.isfinite(y).all()
    print("CPU_FORWARD", tuple(y.shape), str(y.dtype), "finite", True)
finally:
    torch.Tensor.cuda = orig_cuda
    torch.nn.Module.cuda = orig_module_cuda

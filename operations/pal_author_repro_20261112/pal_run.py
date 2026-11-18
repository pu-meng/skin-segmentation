#!/usr/bin/env python3
"""Run the PAL author training loop with fixed project ISIC splits."""
from __future__ import annotations

import contextlib
import importlib
import json
import os
import random
import sys
import time
import types
from pathlib import Path

BASE = Path("/home/PuMengYu/experiments/skin_segmentation")
OP = BASE / "operations/pal_author_repro_20261112"
VENDOR = OP / "vendor/PAL"
DEPS = OP / "vendor_deps"
DATA = Path("/media/8T/PuMengYu/experiments/skin_segmentation/PAL_AUTHOR_ISIC2018_DATA_20261112")


def install_optional_shims() -> None:
    # PAL sets wandb=False by default; keep all logging local and never contact W&B.
    wandb = types.ModuleType("wandb")
    wandb.init = lambda **kwargs: None
    wandb.log = lambda *args, **kwargs: None
    wandb.finish = lambda *args, **kwargs: None
    sys.modules["wandb"] = wandb

    # Apex is absent on the server. Use the source loop in FP32 without altering
    # the architecture or optimizer; this precision deviation is recorded in protocol.
    apex = types.ModuleType("apex")

    class AMP:
        @staticmethod
        def initialize(model, optimizer, **kwargs):
            return model, optimizer

        @staticmethod
        @contextlib.contextmanager
        def scale_loss(loss, optimizer):
            yield loss

    apex.amp = AMP()
    sys.modules["apex"] = apex


def _preflight(trainer, train_module) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from utils.losses import structure_loss, intra_att_loss

    rng_py = random.getstate()
    rng_np = np.random.get_state()
    rng_cpu = torch.get_rng_state()
    rng_cuda = torch.cuda.get_rng_state_all()
    model = trainer.model
    old_mode = model.training
    model.eval()  # avoid changing BatchNorm running statistics during the dry check
    trainer.optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    try:
        image, mask = next(iter(trainer.loader))
        image = image.cuda().float()
        mask = mask.cuda().float().squeeze(1)
        rate = 1.25
        size = int(round(trainer.args.patchsize * rate / 32) * 32)
        image = F.interpolate(image, size=(size, size), mode="bilinear", align_corners=True)
        mask = F.interpolate(mask.unsqueeze(1), size=(size, size), mode="bilinear", align_corners=True).squeeze(1)

        out = model(image, mask)
        att_m = out["att_m"]
        att_v = out["att_v"]
        if trainer.args.decorrtype == "queryfeat":
            query_feat = model.mask2former.query_feat
            feat_corr = torch.cat([m.weight.unsqueeze(0) for m in query_feat], dim=0)
            gram = torch.einsum("ckd,cqd->ckq", feat_corr, feat_corr)
        else:
            att_flat = torch.flatten(att_m, start_dim=2, end_dim=3)
            gram = torch.einsum("bdck,bdcq->bckq", att_flat, att_flat)
        loss_div = intra_att_loss(gram) if trainer.args.intra_div else image.new_zeros(())
        route = out["route_prob"]
        topk = out["topk_mask"]
        loss_balance = train_module.cv_squared(topk.sum(0))
        cls = F.interpolate(out["cls_prob"], size=mask.shape[1:], mode="bilinear", align_corners=True)
        loss_cls = structure_loss(cls, mask, att="interior")
        mcs = out["mcs_prob"]
        if trainer.args.num_classes == 1:
            mcs = mcs[:, 0]
        loss_mcs = structure_loss(mcs, mask, att="boundary")
        loss_fuse = structure_loss(out["fuse_prob"], mask, att="boundary")
        loss = (trainer.args.clsw * loss_cls + trainer.args.mcsw * loss_mcs
                + trainer.args.fusew * loss_fuse + trainer.args.intra_div * loss_div
                + trainer.args.loadbal * loss_balance)
        if not torch.isfinite(loss):
            raise RuntimeError(f"PAL preflight loss non-finite: {loss.item()}")
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not grads or any(not torch.isfinite(g).all() for g in grads):
            raise RuntimeError("PAL preflight gradients missing or non-finite")
        nonzero = sum(int(torch.count_nonzero(g).item()) for g in grads)
        if not nonzero:
            raise RuntimeError("PAL preflight produced no nonzero gradients")
        result = {
            "batch": int(image.shape[0]), "input": list(image.shape),
            "loss": float(loss.detach().item()), "grad_tensors": len(grads),
            "nonzero_grad_elements": nonzero,
            "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1048576, 1),
            "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 1048576, 1),
        }
        print("PAL_PREFLIGHT_OK " + json.dumps(result, sort_keys=True), flush=True)
        trainer.optimizer.zero_grad(set_to_none=True)
        return result
    finally:
        trainer.optimizer.zero_grad(set_to_none=True)
        model.train(old_mode)
        random.setstate(rng_py)
        np.random.set_state(rng_np)
        torch.set_rng_state(rng_cpu)
        torch.cuda.set_rng_state_all(rng_cuda)


def main() -> None:
    physical_gpu = os.environ.get("PAL_PHYSICAL_GPU")
    out_dir = os.environ.get("PAL_ATTEMPT_DIR")
    if physical_gpu not in {"0", "1"} or not out_dir:
        raise SystemExit("PAL_PHYSICAL_GPU and PAL_ATTEMPT_DIR are required")
    os.environ["CUDA_VISIBLE_DEVICES"] = physical_gpu
    os.environ["WANDB_MODE"] = "disabled"
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    sys.path.insert(0, str(DEPS))
    sys.path.insert(0, str(OP / "compat"))
    sys.path.insert(0, str(VENDOR))
    sys.path.insert(0, str(VENDOR / "src"))
    os.chdir(VENDOR / "src")
    install_optional_shims()

    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from types import SimpleNamespace
    from utils.dataset import ISIC2018Data
    from our_model import Model
    train_module = importlib.import_module("our_train")

    args = SimpleNamespace(
        seed=1337, deterministic=1, savepath=out_dir, mode="train", dataset="isic2018",
        datapath=str(DATA), exp="PAL_ISIC2018_", patchsize=352,
        bkbonelr=1e-4, headlr=1e-4, epoch=100, preIter=1000, backbone="pvt",
        batch_size=16, weight_decay=5e-4, opt="Adam", momentum=0.9, nesterov=True,
        snapshot=None, shareW=True, cls_attributes=10, topk=3, num_classes=2,
        num_queries=2, dec_layers=9, tau=1.0, mask_dim=256, sample_num=15,
        fuse_mode="conv", clsw=1.0, mcsw=2.0, fusew=2.0, intra_div=1.0,
        loadbal=1.0, decorrtype="queryfeat", T_max=100, eta_min_values=[0, 0],
        eval=True, evalIter=500, tta=False, postprecessing=False, device="0",
        proj="PAL_local_reproduction", wandb=False,
    )
    args.T_max = args.epoch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train_data = ISIC2018Data(datapath=str(DATA), split="train", patchsize=args.patchsize)
    val_data = ISIC2018Data(datapath=str(DATA), split="valid", patchsize=args.patchsize)
    if len(train_data) != 1780 or len(val_data) != 258:
        raise RuntimeError(f"Unexpected split sizes train={len(train_data)} val={len(val_data)}")
    train_loader = DataLoader(train_data, batch_size=args.batch_size, pin_memory=True,
                              shuffle=False, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=1, pin_memory=True,
                            shuffle=False, num_workers=0)
    train_module.args = args
    train_module.test_loader = val_loader

    trainer = train_module.Train(train_loader, Model, args)
    pvt_weight = VENDOR / "contrastive_model/lib/pvt_v2_b2.pth"
    checkpoint = torch.load(pvt_weight, map_location="cpu", weights_only=True)
    backbone_state = trainer.model.bkbone.state_dict()
    matched = [k for k, value in backbone_state.items()
               if k in checkpoint and value.shape == checkpoint[k].shape]
    if len(matched) != len(backbone_state) or not all(
        torch.equal(backbone_state[k].detach().cpu(), checkpoint[k]) for k in matched
    ):
        raise RuntimeError("PVTv2-B2 pretrained weight verification failed")
    parameters = sum(p.numel() for p in trainer.model.parameters())
    if parameters != 40_264_627:
        raise RuntimeError(f"PAL architecture parameter count drift: {parameters}")
    print(f"PAL_MODEL parameters={parameters} pvt_keys={len(matched)}/{len(backbone_state)}", flush=True)
    preflight = _preflight(trainer, train_module)
    print("PAL_PRECISION FP32_APEX_O2_UNAVAILABLE", flush=True)

    original_train = trainer.train
    train_module.torch = torch
    optimizer = trainer.optimizer
    original_step = optimizer.step
    marker = {"written": False}

    def step_with_marker(*step_args, **step_kwargs):
        result = original_step(*step_args, **step_kwargs)
        if not marker["written"]:
            marker["written"] = True
            print("PAL_FIRST_OPTIMIZER_STEP_OK peak_allocated_mib="
                  + str(round(torch.cuda.max_memory_allocated() / 1048576, 1)), flush=True)
        return result

    optimizer.step = step_with_marker
    torch.cuda.reset_peak_memory_stats()
    original_train()
    if not marker["written"]:
        raise RuntimeError("Training ended without a verified optimizer step")
    print("PAL_TRAINING_COMPLETED", flush=True)


if __name__ == "__main__":
    main()

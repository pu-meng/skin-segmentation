"""Synthetic acceptance checks for A1-swiftformer-LGD-v1.2; never trains or reads test data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from Models.A1SwiftFormerLGD.model import STAGE_CONFIGS
from source.a1_swiftformer_lgd_v12_20261113.candidate import build_candidate


def finite(tensor: torch.Tensor, label: str) -> None:
    if not torch.isfinite(tensor).all():
        raise AssertionError(f"nonfinite values in {label}")


def validate_stage_shapes_and_gradients(model, device: torch.device) -> dict:
    records = {}
    stage_indices = {"D0": 0, "D1": 1, "D2": 2, "D3": 3}
    spatial = {"D0": 64, "D1": 32, "D2": 16, "D3": 8}
    for name, index in stage_indices.items():
        stage = model.decoder.stages[index].to(device)
        cfg = STAGE_CONFIGS[name]
        records[name] = {}
        for batch in (1, 8):
            stage.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            x = torch.randn(batch, cfg.in_channels, spatial[name], spatial[name],
                            device=device, requires_grad=True)
            seen_protection_inputs = []
            handle = stage.block.protection_norm.register_forward_pre_hook(
                lambda _module, args: seen_protection_inputs.append(args[0])
            )
            y = stage(x)
            handle.remove()
            expected = (batch, cfg.channels, spatial[name], spatial[name])
            if tuple(y.shape) != expected:
                raise AssertionError(f"{name} batch {batch}: {tuple(y.shape)} != {expected}")
            finite(y, f"{name} output batch {batch}")
            if not torch.equal(seen_protection_inputs[0], stage.project(x)):
                raise AssertionError(f"{name} protection branch did not read projected raw U")
            y.float().square().mean().backward()
            for label, module in (
                ("entry", stage.block.entry),
                ("global_mamba", stage.block.global_branch.mixer),
                ("local_dw3", stage.block.local_branch.dw3),
                ("local_dw7", stage.block.local_branch.dw7),
                ("protection", stage.block.protection_pw),
                ("fusion", stage.block.fuse_projection),
                ("exit", stage.block.exit_project),
            ):
                grads = [p.grad for p in module.parameters() if p.requires_grad]
                if not grads or any(g is None or not torch.isfinite(g).all() for g in grads):
                    raise AssertionError(f"{name} {label} has missing/nonfinite gradients")
            if stage.block.alpha.grad is None or stage.block.gamma.grad is None:
                raise AssertionError(f"{name} alpha/gamma gradients are missing")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            records[name][str(batch)] = {
                "shape": list(y.shape),
                "finite": True,
                "backward_finite": True,
                "protection_reads_U": True,
                "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)
                if device.type == "cuda" else None,
            }
            del x, y
            stage.zero_grad(set_to_none=True)
    return records


def validate_full_model(model, device: torch.device) -> dict:
    model = model.to(device).train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    batches = {}
    for batch in (1, 8):
        model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        x = torch.rand(batch, 3, 256, 256, device=device)
        y = model(x)
        expected = (batch, 1, 256, 256)
        if tuple(y.shape) != expected:
            raise AssertionError(f"full model output shape is {tuple(y.shape)} != {expected}")
        finite(y, f"full model output batch {batch}")
        y.float().square().mean().backward()
        missing = [i for i, p in enumerate(trainable) if p.grad is None]
        bad = [i for i, p in enumerate(trainable) if p.grad is not None and not torch.isfinite(p.grad).all()]
        if bad:
            raise AssertionError(f"full model has nonfinite gradients at parameter indexes {bad[:8]}")
        if len(missing) > len(trainable) // 20:
            raise AssertionError(f"too many trainable parameters without gradients: {len(missing)}")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        batches[str(batch)] = {
            "output_shape": list(y.shape),
            "output_finite": True,
            "backward_finite": True,
            "missing_gradient_tensors": len(missing),
            "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)
            if device.type == "cuda" else None,
        }
        del x, y
    return {
        "batches": batches,
        "trainable_parameter_tensors": len(trainable),
        "parameters": sum(p.numel() for p in model.parameters()),
        "stage_parameters": {
            stage.config.name: sum(p.numel() for p in stage.parameters())
            for stage in model.decoder.stages
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(2)

    started = time.time()
    model = build_candidate(pretrained=True)
    if type(model.decoder.e2).__name__ != "Identity":
        raise AssertionError("E2A was not removed")
    if type(model.decoder.e3).__name__ != "E3":
        raise AssertionError("original dual-source E3 skip was modified")
    if [type(s).__name__ for s in model.decoder.stages] != ["LGDStage"] * 4:
        raise AssertionError("not all four stages use the shared LGDStage implementation")

    report = {
        "model": "A1-swiftformer-LGD-v1.2",
        "device": str(device),
        "synthetic_only": True,
        "test_data_accessed": False,
        "encoder_loading_audit": model.encoder.loading_audit,
        "stages": validate_stage_shapes_and_gradients(model, device),
        "full_model": validate_full_model(model, device),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    output = ROOT / "operations/a1_swiftformer_lgd_v12_20261113/acceptance_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"ACCEPTANCE_REPORT={output}", flush=True)


if __name__ == "__main__":
    main()

"""Reusable experiment primitives for paper-specific segmentation adapters.

Keep architecture, preprocessing, losses, optimizer, and selection rules in each
paper adapter. This module only owns mechanics that should behave identically.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
from torch import nn


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def extract_logits(output: Any) -> torch.Tensor:
    """Normalize common model returns; adapters should select unusual outputs."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        for key in ("out", "logits", "prediction"):
            if key in output and isinstance(output[key], torch.Tensor):
                return output[key]
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError("Cannot find segmentation logits; provide an adapter forward_logits callback")


def mirror4_logits(
    model: nn.Module,
    images: torch.Tensor,
    forward_logits: Callable[[nn.Module, torch.Tensor], torch.Tensor] = lambda m, x: extract_logits(m(x)),
) -> torch.Tensor:
    """Average logits for identity, horizontal, vertical, and HV mirrored views."""
    views = ((False, False), (True, False), (False, True), (True, True))
    outputs = []
    for horizontal, vertical in views:
        view = images.flip((-1,) if horizontal else ()) if horizontal else images
        if vertical:
            view = view.flip((-2,))
        logits = forward_logits(model, view)
        if vertical:
            logits = logits.flip((-2,))
        if horizontal:
            logits = logits.flip((-1,))
        outputs.append(logits)
    return torch.stack(outputs, dim=0).mean(dim=0)


def binary_global_counts(prediction: torch.Tensor, target: torch.Tensor) -> tuple[int, int, int, int]:
    prediction = prediction.bool()
    target = target.bool()
    return (
        int((prediction & target).sum().item()),
        int((prediction & ~target).sum().item()),
        int((~prediction & target).sum().item()),
        int((~prediction & ~target).sum().item()),
    )


def dice_from_counts(counts: Iterable[int]) -> float:
    tp, fp, fn, _ = (int(value) for value in counts)
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 1.0


def load_checkpoint_strict(model: nn.Module, checkpoint: str | Path, device: torch.device | str) -> nn.Module:
    try:
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only argument
        payload = torch.load(checkpoint, map_location=device)
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in payload and isinstance(payload[key], dict):
                payload = payload[key]
                break
    model.load_state_dict(payload, strict=True)
    return model


def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device | str,
    forward_logits: Callable[[nn.Module, torch.Tensor], torch.Tensor] = lambda m, x: extract_logits(m(x)),
    grad_clip_norm: float | None = None,
    after_step: Callable[[], None] | None = None,
) -> dict[str, float]:
    """Shared supervised epoch loop; paper-specific loss and data stay injectable."""
    model.train()
    losses: list[float] = []
    for batch in loader:
        images, masks = batch[:2]
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = forward_logits(model, images)
        loss = loss_fn(logits, masks)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        if after_step is not None:
            after_step()
        losses.append(float(loss.detach().item()))
    if not losses:
        raise RuntimeError("Training loader produced no batches")
    return {"loss": float(np.mean(losses)), "steps": float(len(losses))}


def evaluate_binary_global(
    model: nn.Module,
    loader,
    device: torch.device | str,
    unpack_batch: Callable[[Any], tuple[torch.Tensor, torch.Tensor, Any]],
    forward_logits: Callable[[nn.Module, torch.Tensor], torch.Tensor] = lambda m, x: extract_logits(m(x)),
    threshold: float = 0.5,
    tta4: bool = False,
) -> dict[str, Any]:
    """Compute pooled-pixel Global Dice and verify unique sample IDs."""
    model.eval()
    total = np.zeros(4, dtype=np.int64)
    ids: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            images, targets, batch_ids = unpack_batch(batch)
            images = images.to(device)
            targets = targets.to(device)
            logits = mirror4_logits(model, images, forward_logits) if tta4 else forward_logits(model, images)
            predictions = torch.sigmoid(logits) >= threshold
            target_masks = targets >= threshold
            total += np.asarray(binary_global_counts(predictions, target_masks), dtype=np.int64)
            if isinstance(batch_ids, (list, tuple)):
                ids.extend(str(value) for value in batch_ids)
            else:
                ids.append(str(batch_ids))
    if not ids:
        raise RuntimeError("Evaluation loader produced no samples")
    if len(set(ids)) != len(ids):
        raise ValueError("Evaluation sample IDs are not unique")
    return {
        "images": len(ids),
        "ids": ids,
        "counts": {"tp": int(total[0]), "fp": int(total[1]), "fn": int(total[2]), "tn": int(total[3])},
        "global_dice": dice_from_counts(total),
        "tta4": bool(tta4),
        "threshold": float(threshold),
    }

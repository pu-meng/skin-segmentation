"""Evaluate all pretrained CPUBone variants on ImageNet validation set.

Runs each model at fp32, bf16, and fp16 and writes the aggregated results to
bf16_fp16_accuracies.txt in the repository root.
"""
import contextlib
import math
import os
import sys

import torch
import torch.utils.data
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from cpubone_model import get_cpubone

# ── Fixed config ──────────────────────────────────────────────────────────────

ROOT = os.path.dirname(os.path.abspath(__file__))

IMAGENET_VAL = (
    "/media/TBData/Datasets/imagenetsmall/imagenet-object-localization-challenge"
    "/ILSVRC/Data/CLS-LOC/val"
)
OUTPUT_FILE = os.path.join(ROOT, "bf16_fp16_accuracies.txt")
BATCH_SIZE = 128
WORKERS = 8
CROP_RATIO = 0.95
IMAGE_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# (display_name, config_path, checkpoint_path, use_ema)
MODELS = [
    # ── Standard CPUBone family ───────────────────────────────────────────────
    (
        "cpubone_nano",
        "configs/cls/imagenet/cpubone_nano.yaml",
        ".exp/cls/imagenet/cpubone_nano/checkpoint/checkpoint.pt",
        False,
    ),
    (
        "cpubone_t0",
        "configs/cls/imagenet/cpubone_t0.yaml",
        ".exp/cls/imagenet/cpubone_t0/checkpoint/evalmodel.pt",
        False,
    ),
    (
        "cpubone_s0",
        "configs/cls/imagenet/cpubone_s0.yaml",
        ".exp/cls/imagenet/cpubone_s0/checkpoint/evalmodel.pt",
        False,
    ),
    (
        "cpubone_b0",
        "configs/cls/imagenet/cpubone_b0.yaml",
        ".exp/cls/imagenet/cpubone_b0/checkpoint/checkpoint.pt",
        False,
    ),
    (
        "cpubone_b1",
        "configs/cls/imagenet/cpubone_b1.yaml",
        ".exp/cls/imagenet/cpubone_b1/checkpoint/evalmodel.pt",
        False,
    ),
    (
        "cpubone_b2",
        "configs/cls/imagenet/cpubone_b2.yaml",
        ".exp/cls/imagenet/cpubone_b2/checkpoint/evalmodel.pt",
        False,
    ),
    (
        "cpubone_b3",
        "configs/cls/imagenet/cpubone_b3.yaml",
        ".exp/cls/imagenet/cpubone_b3/checkpoint/evalmodel.pt",
        False,
    ),
#    ── bf16/fp16-robust re-trainings (full BatchNorm in MBConv) ─────────────
    (
        "cpubone_b0_bftraining",
        "configs/cls/imagenet/cpubone_b0_bftraining.yaml",
        ".exp/cls/imagenet/cpubone_b0_robust/checkpoint/checkpoint.pt",
        True,
    ),
    (
        "cpubone_b1_bftraining",
        "configs/cls/imagenet/cpubone_b1_bftraining.yaml",
        ".exp/cls/imagenet/cpubone_b1_robust/checkpoint/checkpoint.pt",
        True,
    ),
    (
        "cpubone_b2_bftraining",
        "configs/cls/imagenet/cpubone_b2_bftraining.yaml",
        ".exp/cls/imagenet/cpubone_b2_robust/checkpoint/checkpoint.pt",
        True,
    ),
    # ── Ablations ─────────────────────────────────────────────────────────────
    (
        "b1_grouping4_smklstag_notrans",
        "configs/cls/imagenet/b1_grouping4_smklstag_notrans.yaml",
        ".exp/cls/imagenet/b1_grouping4_smklstag_notrans/checkpoint/checkpoint.pt",
        False,
    ),
    (
        "b1_grouping_smk_laststages",
        "configs/cls/imagenet/b1_grouping_smk_laststages.yaml",
        ".exp/cls/imagenet/b1_grouping_smk_laststages/checkpoint/checkpoint.pt",
        False,
    ),
    (
        "b1_grouping_smklstag_notrans_justmbconv",
        "configs/cls/imagenet/b1_grouping_smklstag_notrans_justmbconv.yaml",
        ".exp/cls/imagenet/b1_grouping_smklstag_notrans_justmbconv/checkpoint/checkpoint.pt",
        False,
    ),
]

MODEL_NAMES = [name for name, _, _, _ in MODELS]

PRECISIONS = [
    ("fp32", None),
    ("bf16", torch.bfloat16),
    ("fp16", torch.float16),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else -1.0


def accuracy(output: torch.Tensor, target: torch.Tensor):
    _, pred = output.topk(5, 1, True, True)
    correct = pred.t().eq(target.view(1, -1).expand_as(pred.t()))
    n = target.size(0)
    top1 = correct[:1].reshape(-1).float().sum().mul_(100.0 / n).item()
    top5 = correct[:5].reshape(-1).float().sum().mul_(100.0 / n).item()
    return top1, top5


def load_model(config_path: str, checkpoint_path: str, use_ema: bool) -> torch.nn.Module:
    return get_cpubone(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        pretrained=True,
        use_ema=use_ema,
    )


def build_val_loader() -> torch.utils.data.DataLoader:
    transform = transforms.Compose([
        transforms.Resize(
            int(math.ceil(IMAGE_SIZE / CROP_RATIO)),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return torch.utils.data.DataLoader(
        datasets.ImageFolder(IMAGENET_VAL, transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=True,
        drop_last=False,
    )


def eval_model(model: torch.nn.Module, loader: torch.utils.data.DataLoader, dtype=None, desc=""):
    model.eval()
    model.to(DEVICE)
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    autocast_ctx = (
        torch.autocast(device_type=DEVICE, dtype=dtype)
        if dtype is not None
        else contextlib.nullcontext()
    )

    with torch.inference_mode(), autocast_ctx:
        for images, labels in tqdm(loader, desc=f"  {desc}", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            output = model(images)
            t1, t5 = accuracy(output, labels)
            n = images.size(0)
            top1_meter.update(t1, n)
            top5_meter.update(t5, n)

    return top1_meter.avg, top5_meter.avg


# ── Table formatting ──────────────────────────────────────────────────────────

def build_table(results: dict) -> str:
    NAME_W = 26
    PARAM_W = 8
    CELL_W = 15

    def cell(t1, t5):
        if t1 is None:
            return f"{'N/A':^{CELL_W}}"
        return f"{t1:6.2f}  {t5:6.2f}"

    prec_names = [p for p, _ in PRECISIONS]

    h1 = f"{'Model':<{NAME_W}}  {'Params':>{PARAM_W}}  |"
    h2 = f"{'':>{NAME_W}}  {'(M)':>{PARAM_W}}  |"
    for pname in prec_names:
        h1 += f"  {pname:^{CELL_W}}  |"
        h2 += f"  {'Top1':>6}  {'Top5':>6}  |"

    sep = "-" * len(h1)
    lines = [sep, h1, h2, sep]

    for name in MODEL_NAMES:
        if name not in results:
            continue
        r = results[name]
        params = r.get("params", -1)
        ps = f"{params:.1f}" if params >= 0 else "N/A"
        row = f"{name:<{NAME_W}}  {ps:>{PARAM_W}}  |"
        for pname in prec_names:
            t1, t5 = r.get(pname, (None, None))
            row += f"  {cell(t1, t5)}  |"
        lines.append(row)

    lines.append(sep)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Device : {DEVICE}")
    print(f"Val set: {IMAGENET_VAL}\n")

    loader = build_val_loader()
    results: dict = {}

    for model_name, config_path, checkpoint_path, use_ema in MODELS:
        print(f"\n{'='*60}")
        print(f"  {model_name}")
        print(f"{'='*60}")

        try:
            model = load_model(config_path, checkpoint_path, use_ema)
        except Exception as e:
            print(f"  [SKIP] {e}")
            continue

        params = sum(p.numel() for p in model.parameters()) / 1e6
        results[model_name] = {"params": params}
        print(f"  Params: {params:.1f}M")

        for pname, dtype in PRECISIONS:
            print(f"  Running {pname} ...", end="", flush=True)
            try:
                t1, t5 = eval_model(model, loader, dtype=dtype, desc=pname)
                results[model_name][pname] = (t1, t5)
                print(f"\r  {pname}: Top1={t1:.2f}  Top5={t5:.2f}")
            except Exception as e:
                print(f"\r  {pname}: ERROR — {e}")
                results[model_name][pname] = (None, None)

        del model
        torch.cuda.empty_cache()

    table = build_table(results)
    print(f"\n\n{table}\n")

    with open(OUTPUT_FILE, "w") as f:
        f.write(table + "\n")
    print(f"Results saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

from typing import Tuple

import torch
from ptflops import get_model_complexity_info
from thop import profile


def describe_model(
    model: torch.nn.Module,
    model_name: str,
    out_size: Tuple[int, int],
    channels: int = 3,
) -> None:
    """
    Print model architecture and compute complexity.

    Args:
        model: The PyTorch model to describe.
        model_name: A human-readable name for the model.
        out_size: Output spatial size as (height, width).
        channels: Number of input channels. Defaults to 3.
    """
    print("-" * 100)
    print(f"Model {model_name} architecture:")
    num_params = sum(p.numel() for p in model.parameters())
    print(model)
    print("-" * 100)

    # Compute model FLOPs and parameters (thop)
    dummy_input = torch.randn(1, channels, out_size[0], out_size[1]).cuda()
    # 注意：inputs 需要传入单元素元组，否则会被当作可迭代对象解包，导致 forward 接收 3D 张量
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)  # pylint: disable=no-member, W0632
    print("\n--- [THOP] Model complexity ---")
    print(f"FLOPs: {flops / 1e9:.2f} G")
    print(f"Parameters: {params / 1e6:.2f} M")

    # Compute model FLOPs and parameters (ptflops)
    flops_pt, params_pt = get_model_complexity_info(
        model, (channels, out_size[0], out_size[1]), as_strings=True, print_per_layer_stat=False
    )
    print("\n--- [ptflops] Model complexity ---")
    print(f"FLOPs: {flops_pt}")
    print(f"Parameters: {params_pt}")

    print(f"\nTotal trainable parameters: {num_params}")
    print("-" * 100)

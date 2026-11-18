"""Build controls at the same insertion sites as existing V2 experiments."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'source/comparison')]
import torch
from torch import nn
from pretrained_unet2d import PretrainedUNet2D, prepare_imports


def build(backbone, kind, pretrained=True, target_parameters=1695744):
    prepare_imports('mambavision_t')
    from Models.TinyViM.bottleneck_parameter_controls import ParameterMatchedBottleneck
    # Match the existing parent initialization; module creation preserves its RNG.
    torch.manual_seed(42)
    if backbone == 'tinyvim':
        model = PretrainedUNet2D('tinyvim_s', pretrained=pretrained, new_layers_seed=42)
        model.align[-1] = nn.Sequential(model.align[-1], ParameterMatchedBottleneck(
            kind, target_parameters=target_parameters, seed=44))
    elif backbone == 'unet':
        from unet2d import UNet2D
        model = UNet2D()
        model.encoders[-1] = nn.Sequential(model.encoders[-1], ParameterMatchedBottleneck(
            kind, target_parameters=target_parameters, seed=44))
    else:
        raise ValueError(backbone)
    return model

"""E1.3: preserve original four DoubleConv stages and append ConvWaveStage."""
from pathlib import Path
import sys
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).parent
sys.path[:0]=[str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
from source.e1_variants.candidate import ConvWaveStage
sys.path.insert(0,str(HERE))


def build_candidate(pretrained=True):
    from Models.TinyViM.four_stage_decoder import build_four_stage_baseline
    model=build_four_stage_baseline(pretrained=pretrained)
    stages=list(model.decoder.stages)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(44)
        model.decoder.stages=nn.ModuleList([
            nn.Sequential(stage,ConvWaveStage(channels,channels,resolution))
            for stage,channels,resolution in zip(stages,[64,128,256,512],[64,32,16,8])
        ])
    return model

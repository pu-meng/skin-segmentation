"""Direct composition of completed E1 and E3, preserving both initializations."""
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).parent
sys.path[:0]=[str(ROOT/'source/three_module_impl_20261104'),str(ROOT),str(ROOT/'source/comparison'),str(ROOT/'source/bottleneck_mamba')]
from source.three_module_impl_20261104.candidate import DStage
from source.three_module_e23_20261104.candidate import build_candidate as build_e3
from Models.TinyViM.four_stage_decoder import build_four_stage_baseline
sys.path.insert(0,str(HERE))

def build_candidate(pretrained=True,arm='E1_E3'):
    if arm=='E0':return build_four_stage_baseline(pretrained=pretrained)
    if arm!='E1_E3':raise ValueError(arm)
    model=build_e3(pretrained=pretrained,arm='E3')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model.decoder.stages[3]=DStage()
    return model

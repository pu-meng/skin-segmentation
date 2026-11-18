"""A1 E+S historical seed42 baseline with only D2 replaced."""
import torch
from Models.A1SwiftFormerGLR.model import D2GLR
from operations.a1swift_pair_ablation_20261112.entry import build as build_baseline

def build_candidate(variant,pretrained=True):
    model=build_baseline('no_e2_no_d2d3',pretrained=pretrained)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261114)
        model.decoder.stages[2]=D2GLR(variant)
    return model

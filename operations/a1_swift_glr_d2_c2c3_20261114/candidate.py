"""Historical E+S baseline; C2/C3 differ from C1 only in refinement."""
import torch
from Models.A1SwiftFormerGLR.model import D2GLR as AnchorD2
from Models.A1SwiftFormerGLRv12.model import D2GLR
from operations.a1swift_pair_ablation_20261112.entry import build as build_baseline

def build_candidate(variant, pretrained=True):
    model=build_baseline('no_e2_no_d2d3',pretrained=pretrained)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261114)
        anchor=AnchorD2('C1')
        torch.manual_seed(20261114)
        replacement=D2GLR(variant)
        source=anchor.state_dict(); target=replacement.state_dict()
        for key in target:
            if not key.startswith('block.refine.'):
                assert key in source and target[key].shape==source[key].shape
                target[key]=source[key].clone()
        replacement.load_state_dict(target,strict=True)
        model.decoder.stages[2]=replacement
    return model

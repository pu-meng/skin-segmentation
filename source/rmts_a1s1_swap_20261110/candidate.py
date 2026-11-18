"""Official pretrained RMT-S encoder on unchanged A1-S1 / A1-S1-SkipV0.6 decoder."""
from operations.encoder_swap_A016_20261107.encoder_swap_candidate import build_candidate as build_rmt_a1

def build_candidate(pretrained=True, variant='s1'):
    if variant == 's1':
        from source.a1_s1_20261108.candidate import build_candidate as build_parent
    elif variant == 'skip':
        from source.a1_s1_skip1_v06_20261108.candidate import build_candidate as build_parent
    else:
        raise ValueError(variant)
    parent=build_parent(pretrained=False)
    rmt=build_rmt_a1('rmt_s__A1',pretrained=pretrained)
    parent.encoder=rmt.encoder
    parent.align=rmt.align
    return parent

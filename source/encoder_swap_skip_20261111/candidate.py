"""Official pretrained SwiftFormer-S / MobileOne-S1 encoder on unchanged A1-S1-SKIPv0.6 decoder.

Same construction as the RMT-S swap (source/rmts_a1s1_swap_20261110/candidate.py), but the
encoder is selectable. The parent (A1-S1-SKIPv0.6) is rebuilt exactly; only `encoder` and the
1x1 channel-alignment layers are replaced by the official pretrained backbone.
"""
from operations.encoder_swap_A016_20261107.encoder_swap_candidate import build_candidate as build_base

ENCODERS = {
    'swiftformer_s': 'SwiftFormer-S',
    'mobileone_s1': 'MobileOne-S1',
}


def build_candidate(pretrained=True, encoder='swiftformer_s'):
    assert encoder in ENCODERS, encoder
    from source.a1_s1_skip1_v06_20261108.candidate import build_candidate as build_parent
    parent = build_parent(pretrained=False)
    base = build_base(f'{encoder}__A1', pretrained=pretrained)
    parent.encoder = base.encoder
    parent.align = base.align
    assert all(p.requires_grad for p in parent.parameters())
    return parent

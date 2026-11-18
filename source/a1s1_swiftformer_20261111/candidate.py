"""Official pretrained SwiftFormer-S encoder on the unchanged A1-S1 decoder (no Skip1-v0.6).

Why this run exists: the locked backbone is A1-S1 + Skip1-v0.6. Swapping its encoder to
SwiftFormer-S scored 90.8189 on ISIC2018, while A1 + SwiftFormer-S scored 91.8683. That
1.05 pp gap conflates two things — the A1 -> A1-S1 structure change and Skip1-v0.6 itself.
This candidate holds Skip1-v0.6 out and keeps everything else, so the gap can be split.

Construction is identical to source/encoder_swap_skip_20261111/candidate.py: the parent
A1-S1 model is rebuilt exactly, then only `encoder` and `align` are replaced by the official
pretrained backbone taken from the A1 arm of the encoder-swap matrix.
"""
from operations.encoder_swap_A016_20261107.encoder_swap_candidate import build_candidate as build_base


def build_candidate(pretrained=True, encoder='swiftformer_s'):
    assert encoder == 'swiftformer_s', encoder
    from source.a1_s1_20261108.candidate import build_candidate as build_parent
    parent = build_parent(pretrained=False)
    base = build_base(f'{encoder}__A1', pretrained=pretrained)
    parent.encoder = base.encoder
    parent.align = base.align
    assert all(p.requires_grad for p in parent.parameters())
    return parent

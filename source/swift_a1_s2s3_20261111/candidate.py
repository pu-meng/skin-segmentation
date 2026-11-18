"""SwiftFormer-S encoder + A1 decoder + S2S3 Skip2/Skip3 modules.

Encoder: official pretrained SwiftFormer-S taken from the encoder-swap matrix, which rebuilds the
A1 decoder unchanged and only replaces `encoder` and `align`. Same construction path that produced
A1 + SwiftFormer-S = 91.8683 ISIC2018 (no TTA).

Decoder: the A1 decoder with Skip2/Skip3 replaced by the S2S3 modules from
source/a1_s2s3_20261111. Everything else (Bottleneck E2A, Decoder3 D-waveformer, Decoder2
DualStage, Decoder1, Skip1 E3, Decoder0, Head) stays as in A1.
"""
import torch

from source.a1_s2s3_20261111.candidate import (  # noqa: F401
    DecoderS2S3,
    SKIP_CHANNELS,
    SkipS2S3,
)


def build_candidate(pretrained=True, use_skip2=True, use_skip3=True,
                    use_gate=True, use_local=True, gate_mode='spatial',
                    impl='mamba', beta=1.0, rho=0.1):
    from operations.encoder_swap_A016_20261107.encoder_swap_candidate import (
        build_candidate as build_swap,
    )

    model = build_swap('swiftformer_s__A1', pretrained=pretrained)
    old = model.decoder
    skip2 = skip3 = None
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4202)
        if use_skip3:
            c, expand = SKIP_CHANNELS['skip3']
            skip3 = SkipS2S3(c, expand, impl, use_gate, use_local, gate_mode, beta, rho)
        torch.manual_seed(4203)
        if use_skip2:
            c, expand = SKIP_CHANNELS['skip2']
            skip2 = SkipS2S3(c, expand, impl, use_gate, use_local, gate_mode, beta, rho)
    model.decoder = DecoderS2S3(old, skip2, skip3)
    return model

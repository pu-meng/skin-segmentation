import os
from candidate import ROOT

ARM = os.environ['E1_VARIANT']
assert ARM in ('E1.1', 'E1.2')
if ARM == 'E1.1':
    from candidate import build_candidate
    RUN = ROOT/'log/TinyViM_E1.1_四层卷积DWPO_HAM_20261104'
    CHOICES = dict(
        placement='replace all four D3/D2/D1/D0 DoubleConv stages',
        pre='DW3(cin groups) -> PW1(cin,C) -> BN -> GELU',
        core='original E1 directional WaveOperator; X+gamma*wave(X); gamma init .001',
        post='DW3(C groups) -> PW1(C,C) -> BN -> GELU',
        stage_interfaces=[[128,64,64], [256,128,32], [512,256,16], [512,512,8]],
        ffn='no additional FFN', lambda_change=False,
    )
else:
    from gated_lambda import build_candidate
    RUN = ROOT/'log/TinyViM_E1.2_输入门控lambda_HAM_20261104'
    CHOICES = dict(
        placement='original E1 D3 only; D2/D1/D0 remain original DoubleConv',
        gate_input='D3 input X, before existing wave DW3',
        gate='GAP -> Conv1x1(512,32) -> GELU -> Conv1x1(32,512) -> 2sigmoid',
        gate_shape='B,512,1,1; image-conditioned; broadcast over frequency grid',
        lambda_formula='lambda_new[b,c,h,w]=g[b,c]*lambda_E1[c,h,w]',
        initialization='gate last weight/bias zero, g=1; all original E1 init tensors retained',
        independence='from original E1, not E1.1; no added surrounding convolutions',
    )


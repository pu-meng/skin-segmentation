from candidate import ROOT,build_candidate
ARM='E1.3'
RUN=ROOT/'log/TinyViM_E1.3_保留DoubleConv_四层追加DWPO_HAM_20261105'
CHOICES=dict(
 placement='retain all four original D3/D2/D1/D0 DoubleConv; append one ConvWaveStage after each',
 stage_order='original DoubleConv(cin,C) -> pre separable Conv(C,C) -> X+gamma*Wave(X) -> post separable Conv(C,C)',
 pre='DW3(C groups) -> PW1(C,C) -> BN -> GELU',
 core='unchanged E1.1 DirectionalWaveOperator and alpha_embed; gamma init .001',
 post='DW3(C groups) -> PW1(C,C) -> BN -> GELU',
 stage_interfaces=[[128,64,64],[256,128,32],[512,256,16],[512,512,8]],
 initialization='official ImageNet encoder; original DoubleConv/head/up tensors preserved at seed42; align43; independent appended modules44',
 ffn='no additional FFN',lambda_change=False,
 comparison='independent E0-based addition experiment; no E1/E1.1 HAM weights loaded; not a one-factor matched-count comparison with E1.1',
)

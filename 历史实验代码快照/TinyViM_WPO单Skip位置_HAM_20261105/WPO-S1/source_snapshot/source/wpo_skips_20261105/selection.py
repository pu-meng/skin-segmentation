import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
ARM=os.environ['WPO_ARM'];assert ARM in ['WPO-S0','WPO-S1','WPO-S2']
LEVEL=int(ARM[-1]);QUEUE=ROOT/'log/TinyViM_WPO单Skip位置_HAM_20261105';RUN=QUEUE/ARM
CHOICES=dict(base='E0 reused, not retrained',placement=f'Align{LEVEL} -> S + gamma*DirectionalWPO(S) -> matching decoder concat only',resolution=64//2**LEVEL,channels=64*2**LEVEL,retained='all E0 stages including D3 DoubleConv, ups/align/head and other skips; no encoder feedback',gamma='per-channel .001',time=1,K=1,alpha='(H,W//2+1,1), inverse_softplus(.05)',direction='independent per channel, ax=1+tanh(raw), ay=1-tanh(raw)',reference='existing E1 replaces D3; this adds a skip branch, not equal-parameter or identical replacement control',excluded='RGB input WPO/E2/E3.1/RAP/all-skips/ISIC/five-seed',initialization='ImageNet encoder, paired E0 decoder42/align43; additional module isolated seed42',scope='HAM seed42 full network FP32 batch8 256x256 300 epochs; earliest min-val-loss best then 2020-image test',gpu='GPU0 authorized shared, maximum two new arms; existing GPU0 B3 reservation completes first')

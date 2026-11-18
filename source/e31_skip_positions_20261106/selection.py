import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
ARM=os.environ['E31_ARM']
assert ARM in ['E31-S1','E31-S2']
LEVEL=int(ARM[-1]);CHANNELS=64*2**LEVEL;RESOLUTION=64//2**LEVEL
QUEUE=ROOT/'log/TinyViM_E3.1单Skip位置_HAM_20261106';RUN=QUEUE/ARM
CHOICES=dict(base='E0 unchanged four-stage decoder',placement=f'up{LEVEL} D and S{LEVEL} -> original E3.1 refiner -> concat(D,Snew) -> original D{LEVEL}',
 level=LEVEL,channels=CHANNELS,resolution=RESOLUTION,
 architecture='original FullScan(C) plus RepDW7(C)/official FDConv3(C,C); independent cross gates; D spatial softmax2; projected D; outer eta.001',
 global_scan=dict(d_state=16,dt_rank=(CHANNELS+15)//16,directions=4),
 initialization='official ImageNet encoder, E0 decoder42/align43; isolated refiner42',
 retained='all original encoder/align/ups/DoubleConv/head and other skips',
 scope='HAM seed42 batch8 256 FP32 full-network 300 epochs original AdamW cosine T_max50; val minloss earliest best then no-TTA test2020',
 S0='reuse audited original E3.1 no-TTA test; verify generalized S0 identical model state and output; no S0 retrain',
 interpretation='position plus natural channel-width change; not equal-parameter ablation',
 gpu='GPU0 shared; existing B3D repeat kept, at most two GPU0 compute jobs; no GPU1 use')

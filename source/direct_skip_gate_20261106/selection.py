from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
ARM='DG-S0'
QUEUE=ROOT/'log/TinyViM_Decoder直接选择Skip_HAM_20261106';RUN=QUEUE/ARM
CHOICES=dict(base='E0 unchanged four-stage decoder',placement='up0 D -> Conv1x1 64to64 -> sigmoid weight -> multiply original S0 -> concat(D,Snew) -> original D0',
 gate='Snew = S * sigmoid(Conv1x1(D)); B64x64x64 weight, pixel and channel dependent',
 initialization='ImageNet encoder; paired E0 decoder42/align43; new gate isolated42; gate weight and bias zero, initial gate .5',
 retained='all original stages, ups, aligns, head and S1/S2',
 excluded='no E3.1 FullScan/FDConv/RepDW, no Skip transform, no D projection/residual scaling',
 scope='HAM seed42 batch8 input256 FP32 full-network 300epochs; original AdamW and cosine T_max50; validation minloss earliest best then independent test2020',
 interpretation='direct attenuation in (0,1), initial Skip amplitude .5; not identity initialization; no guaranteed metric benefit',
 gpu='GPU0 after E31-S1/S2 complete train/test; preserve B3D and GPU1 jobs')

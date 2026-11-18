from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
ARM='B3-E1-E3.1'
QUEUE=ROOT/'log/TinyViM_B3_E1_E3.1_HAM_20261105'
RUN=QUEUE/ARM
CHOICES={'placement': 'original Align3 -> B3 three independent full-width Mamba residuals -> original E1 DStage replacing D3 -> original D2/D1 -> up0 D and original S0 -> original E3.1 DynamicSkip(S,D) -> concat(D,Snew) -> original D0/head', 'initialization': 'ImageNet encoder, paired E0 decoder42 align43; B3 isolated44, E1 isolated42, E3.1 isolated42; no HAM weights', 'B3': 'LN512 Mamba1 state16 conv4 expand2, row-major64 tokens, three nonshared external residuals', 'E1': 'full original directional WPO t1 K1 gamma.001 alpha8x5x1 invsoftplus(.05)', 'E31': 'full Models/E3DynamicSkip/model.py DynamicSkip; FullScan64 FDConv RepDW7 two cross gates decoder spatial softmax project eta.001', 'scope': 'HAM6983/1010/2020 seed42 full-network FP32 batch8 input256 300epochs AdamW.001 decay.01 cosine T_max50; earliest min-val-loss fixed-best strict test', 'gpu': '2026-11-06 user requests start now; GPU0 shared using free memory, retain B3D/E31 jobs and pending reservations; no GPU1 use'}

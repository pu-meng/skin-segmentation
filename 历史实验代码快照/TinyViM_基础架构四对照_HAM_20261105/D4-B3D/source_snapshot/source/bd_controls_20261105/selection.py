import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
ARM=os.environ['BD_ARM']
ARMS={'D4-B1':(1,False),'D4-B3':(3,False),'D4-BD':(0,True),'D4-B3D':(3,True)}
assert ARM in ARMS
COUNT,MED=ARMS[ARM]
QUEUE=ROOT/'log/TinyViM_基础架构四对照_HAM_20261105'
RUN=QUEUE/ARM
CHOICES=dict(base='E0 reused, no retraining',placement='Align3 -> independent Mamba residual blocks -> original D3 or MedNeXt D3',mamba=dict(class_name='official mamba_ssm.modules.mamba_simple.Mamba',scan='one forward row-major 1D; no SS2D/no E2',blocks=COUNT,shared_parameters=False,d_model=512,d_state=16,d_conv=4,expand=2,dt_rank='auto=32',outer='tokens + Mamba(LayerNorm(tokens))',seed=44),mednext=dict(enabled=MED,source='installed official nnunet_mednext MedNeXtBlock, exact vendored blocks.py',dim='2d',kernel_size=7,exp_r=4,blocks_per_stage=2,norm_type='group',groups='one per channel',do_res=True,grn=False,concat_adapter='Conv2d1x1 input -> output then two equal-width official residual blocks; D0 128->64,D1 256->128,D2 512->256,D3 512->512 identity',retained='original transposed-conv ups, skip concat, alignment, head'),scope='HAM seed42 batch8 256x256 300 epochs full network; val minloss earliest best; fixed-best 2020-image test; no E1/E2/E3, ISIC or five seed',gpu='GPU0 shared; two at a time, prior jobs untouched',initialization='ImageNet encoder, E0 decoder42/align43; added modules isolated seed44')

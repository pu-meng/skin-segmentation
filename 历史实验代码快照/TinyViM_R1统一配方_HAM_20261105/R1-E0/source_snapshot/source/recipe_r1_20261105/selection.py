import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
ARM=os.environ['R1_ARM']
COUNTS={'R1-E0':0,'R1-B3':3,'R1-B1':1}
assert ARM in COUNTS
COUNT=COUNTS[ARM]
QUEUE=ROOT/'log/TinyViM_R1统一配方_HAM_20261105'
RUN=QUEUE/ARM
CHOICES=dict(placement='unchanged E0 or original BasicMambaResidual x count before original D3',count=COUNT,med=False,initialization='ImageNet encoder; paired decoder42 align43; Mamba44 fork RNG; no HAM weights')

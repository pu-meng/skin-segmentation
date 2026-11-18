from candidate import ROOT,build_candidate
ARM='E2.2'
RUN=ROOT/'log/TinyViM_E2.2_新版全局局部_HAM_20261105'
CHOICES={'placement': 'Align3 -> latest E2 -> retained original D3 DoubleConv', 'switches': {'E1': False, 'E2': True, 'E3': False}, 'E2_source': 'read-only frozen source/e123_joint_20261105/candidate.py:E2', 'definition': 'latest module plan chapter2 including rho_G/L initial .1 and outer gamma .01', 'FFN': 'same documented first implementation choice as E123.1: TinyViM ConvBN256->1024 GELU ConvBN1024->256, drop0, separate preNorm', 'GPU': 'physical GPU1 shared with E123.1 and existing jobs, do not terminate existing processes', 'scope': 'HAM300 seed42 only; no ISIC/five seed', 'initialization': 'ImageNet encoder and original E0 decoder42/align43 retained; E2 independent seed42'}

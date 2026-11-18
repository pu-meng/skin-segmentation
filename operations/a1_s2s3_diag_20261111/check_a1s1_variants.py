import sys
sys.path.insert(0, '/home/PuMengYu/experiments/skin_segmentation')
sys.path[:0] = ['/home/PuMengYu/experiments/skin_segmentation/source/comparison',
                '/home/PuMengYu/experiments/skin_segmentation/source/bottleneck_mamba',
                '/home/PuMengYu/experiments/skin_segmentation/source/three_module_impl_20261104']
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t')
prepare_imports('tinyvim_s')

from source.a1s1_skipv06_ablate_unet_20261111.candidate import build_candidate

for variant in ('none', 'no_skip1', 'no_e2', 'no_d2d3'):
    m = build_candidate(pretrained=False, variant=variant)
    d = m.decoder.base
    n = sum(p.numel() for p in m.parameters())
    print(f'{variant:9s} params={n:,} e2={type(d.e2).__name__} e3={type(d.e3).__name__} '
          f'st2={type(d.stages[2]).__name__} st3={type(d.stages[3]).__name__}')

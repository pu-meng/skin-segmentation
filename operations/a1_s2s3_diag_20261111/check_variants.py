import sys
sys.path.insert(0, '/home/PuMengYu/experiments/skin_segmentation')
sys.path[:0] = ['/home/PuMengYu/experiments/skin_segmentation/source/comparison',
                '/home/PuMengYu/experiments/skin_segmentation/source/bottleneck_mamba',
                '/home/PuMengYu/experiments/skin_segmentation/source/three_module_impl_20261104']
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t')
prepare_imports('tinyvim_s')

from source.a1_swift_ablate_20261111.candidate import build_candidate

for variant in ('none', 'no_skip1', 'no_e2', 'no_d2d3'):
    m = build_candidate(pretrained=False, variant=variant)
    n = sum(p.numel() for p in m.parameters())
    e3 = type(m.decoder.e3).__name__
    st2 = type(m.decoder.stages[2]).__name__
    st3 = type(m.decoder.stages[3]).__name__
    e2 = type(m.decoder.e2).__name__
    print(f'{variant:9s} params={n:,} e2={e2} e3={e3} stages2={st2} stages3={st3}')

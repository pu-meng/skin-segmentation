import sys
sys.path.insert(0, '/home/PuMengYu/experiments/skin_segmentation')
sys.path[:0] = ['/home/PuMengYu/experiments/skin_segmentation/source/comparison',
                '/home/PuMengYu/experiments/skin_segmentation/source/bottleneck_mamba',
                '/home/PuMengYu/experiments/skin_segmentation/source/three_module_impl_20261104']
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t')
prepare_imports('tinyvim_s')

from source.a1_s1_skip1_v06_20261108.candidate import build_candidate as build_parent
from Models.TinyViM.four_stage_decoder import build_four_stage_baseline

m = build_parent(pretrained=False)
d = m.decoder
print('decoder', type(d).__name__, 'params', f'{sum(p.numel() for p in m.parameters()):,}')
print('e2', type(d.e2).__name__, 'e3', type(d.e3).__name__)
print('stages2', type(d.stages[2]).__name__, 'stages3', type(d.stages[3]).__name__)
import inspect
print('e3.forward sig', inspect.signature(d.e3.forward))
b = build_four_stage_baseline(pretrained=False)
print('base stages2', type(b.decoder.stages[2]).__name__, 'stages3', type(b.decoder.stages[3]).__name__)

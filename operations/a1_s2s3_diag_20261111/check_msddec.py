import sys
sys.path.insert(0, '/home/PuMengYu/experiments/skin_segmentation')
sys.path[:0] = ['/home/PuMengYu/experiments/skin_segmentation/source/comparison',
                '/home/PuMengYu/experiments/skin_segmentation/source/bottleneck_mamba',
                '/home/PuMengYu/experiments/skin_segmentation/source/three_module_impl_20261104']
import torch
from pretrained_unet2d import prepare_imports
prepare_imports('mambavision_t')
prepare_imports('tinyvim_s')

from source.a1_swift_msddec_20261111.candidate import build_candidate

for variant in ('none', 'msddec'):
    m = build_candidate(pretrained=False, variant=variant)
    n = sum(p.numel() for p in m.parameters())
    m = m.cuda().eval()
    x = torch.randn(1, 3, 256, 256, device='cuda')
    with torch.no_grad():
        y = m(x)
    print(f'{variant:7s} params={n:,} out={tuple(y.shape)} range=({float(y.min()):.3f},{float(y.max()):.3f})',
          flush=True)
    del m, x, y
    torch.cuda.empty_cache()

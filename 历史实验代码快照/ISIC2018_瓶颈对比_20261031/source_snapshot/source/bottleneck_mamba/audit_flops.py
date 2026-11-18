"""CPU shape/parameter audit and analytical major-op FLOPs; no checkpoints/data.

MAC=2 FLOPs. Scan uses an explicitly approximate 8/7 coefficient (exp proxy
included); this is not an exact count of executed CUDA instructions.
"""
import hashlib
import inspect
import json
import math
from collections import defaultdict
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT), str(PROJECT/'source/comparison'), str(Path(__file__).parent)]
import torch
from torch import nn
from final_test_pipeline import build_model, ARMS
from pretrained_unet2d import prepare_imports

OUT = PROJECT/'研究记录'


def main():
    torch.set_num_threads(2)
    prepare_imports('mambavision_t')
    from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
    from mamba_ssm.modules.mamba_simple import Mamba
    rows = {}
    sources = {Path(__file__), Path(inspect.getfile(Mamba)),
               Path(inspect.getfile(selective_scan_ref))}
    sources.update(Path(__file__).parent/name for name in (
        'final_test_pipeline.py', 'experiment_v1.py', 'experiment_extra.py', 'experiment_unet_v2.py'))
    sources.add(PROJECT/'source/comparison/pretrained_unet2d.py')
    for arm in ('tinyvim_base', 'tinyvim_v1', 'tinyvim_v2', 'unet_base', 'unet_v1', 'unet_v2'):
        model = build_model(arm).eval()
        # CPU-only mathematical reference adapter, scoped to this audit process.
        # Same recurrence and dimensions; no on-disk source/package changes.
        if arm.startswith('tinyvim'):
            ss = next(m for m in model.modules() if type(m).__name__ == 'SS2D')
            module = sys.modules[type(ss).__module__]
            module.SelectiveScan.apply = lambda u, delta, A, B, C, D, bias, softplus, nrows: selective_scan_ref(
                u, delta, A, B, C, D, delta_bias=bias, delta_softplus=softplus)
        entries, handles = [], []
        def add(name, kind, flops, **detail):
            entries.append(dict(name=name, kind=kind, flops=int(flops), **detail))
        for name, m in model.named_modules():
            if type(m).__module__ != 'torch.nn.modules.container':
                try:
                    sources.add(Path(inspect.getfile(type(m))))
                except (TypeError, OSError):
                    pass
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                def hook(m, args, output, name=name):
                    if isinstance(m, nn.Linear):
                        f = 2 * output.numel() * m.in_features
                    elif isinstance(m, nn.ConvTranspose2d):
                        f = 2 * args[0].numel() * (m.out_channels//m.groups) * math.prod(m.kernel_size)
                    else:
                        f = 2 * output.numel() * (m.in_channels//m.groups) * math.prod(m.kernel_size)
                    add(name, 'conv_linear', f, input=list(args[0].shape), output=list(output.shape))
                handles.append(m.register_forward_hook(hook))
            elif type(m).__name__ == 'SS2D':
                def hook(m, args, output, name=name):
                    b, _, h, w = args[0].shape
                    pool = 2**(3-m.index)
                    l = (h//pool)*(w//pool)
                    c, n, r, k = m.split[0], m.d_state, m.dt_rank, m.K
                    add(name, 'ss2d_parameter_projections', 2*b*k*l*c*(2*r+2*n),
                        batch=b, length=l, channels=c, state=n, rank=r, directions=k)
                    add(name, 'scan_approx', 8*b*k*l*c*n)
                handles.append(m.register_forward_hook(hook))
            elif type(m).__name__ in ('DirectParameterMamba', 'NoDeltaInjectionMamba'):
                def hook(m, args, output, name=name):
                    b, l, d = args[0].shape
                    coeff = 8 if type(m).__name__ == 'DirectParameterMamba' else 7
                    add(name, 'scan_approx', coeff*b*l*m.d_inner*m.d_state,
                        batch=b, length=l, model_dim=d, inner=m.d_inner,
                        state=m.d_state, coefficient=coeff)
                handles.append(m.register_forward_hook(hook))
        with torch.inference_mode():
            y = model(torch.zeros(1, 3, 256, 256))
        assert tuple(y.shape) == (1, 1, 256, 256) and torch.isfinite(y).all()
        for h in handles:
            h.remove()
        categories = defaultdict(int)
        for e in entries:
            categories[e['kind']] += e['flops']
        params = sum(p.numel() for p in model.parameters())
        provenance = ARMS[arm][1]/'provenance.json'
        assert params == json.loads(provenance.read_text())['parameters']
        rows[arm] = dict(parameters=params, major_flops=sum(categories.values()),
                         categories=dict(categories), operations=entries,
                         parameter_check=str(provenance))
        print(arm, params, dict(categories), 'GFLOPs', rows[arm]['major_flops']/1e9, flush=True)
    for base in ('tinyvim', 'unet'):
        for version in ('v1', 'v2'):
            row = rows[f'{base}_{version}']
            baseline = rows[f'{base}_base']
            inserted = 'align.3.1.' if base == 'tinyvim' else 'encoders.4.1.'
            added = sum(e['flops'] for e in row['operations'] if e['name'].startswith(inserted))
            assert row['major_flops'] == baseline['major_flops'] + added
            row['added_major_flops'] = added
    files = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources) if p.is_file()}
    result = dict(input=[1,3,256,256], mode='CPU eval/inference; random weights; no dataset/checkpoint',
        convention='MAC=2; Conv/Linear and SS2D einsum counted; scan approx coefficient 8, V2 7',
        omitted=['bias', 'normalization', 'activation/softplus/sigmoid', 'pooling',
                 'interpolation', 'residual/cross-merge adds', 'D skip and z gate', 'memory operations'],
        causal_convolution='Count full padded output before crop: kernel4 L+3, generator kernel3 L+2',
        scan_caveat='8/7 are pedagogical approximations including exp as a unit proxy; contraction/fusion can change actual arithmetic',
        source_sha256=files, models=rows)
    (OUT/'基础_版本1_版本2_FLOPs核对.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()

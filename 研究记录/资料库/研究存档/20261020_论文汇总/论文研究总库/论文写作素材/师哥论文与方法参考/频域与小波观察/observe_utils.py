"""Read-only CT / Haar / frozen-network observation helpers. No GPU use."""
from pathlib import Path
from itertools import product
import hashlib
import json
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

LABEL_CMAP = ListedColormap(['black', '#20c997', '#ff4757'])
LABEL_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def haar(a):
    """Orthonormal decimated Haar along all spatial axes; even sizes required.

    Keys follow NumPy array axes, e.g. z,y,x for 3D; H=(even-odd)/sqrt(2).
    """
    a = np.asarray(a, dtype=np.float32)
    if any(n % 2 for n in a.shape):
        raise ValueError(f'Haar demo requires even dimensions, received {a.shape}')
    bands = {'': a}
    for axis in range(a.ndim):
        result = {}
        for name, value in bands.items():
            even = np.take(value, np.arange(0, value.shape[axis], 2), axis=axis)
            odd = np.take(value, np.arange(1, value.shape[axis], 2), axis=axis)
            result[name+'L'] = (even + odd) / np.sqrt(2.)
            result[name+'H'] = (even - odd) / np.sqrt(2.)
        bands = result
    return bands


def ihaar(bands):
    """Inverse of haar, preserving signed detail coefficients."""
    current = dict(bands)
    ndim = len(next(iter(current)))
    for axis in reversed(range(ndim)):
        result = {}
        for prefix in sorted({name[:-1] for name in current}):
            low, high = current[prefix+'L'], current[prefix+'H']
            shape = list(low.shape)
            shape[axis] *= 2
            out = np.empty(shape, dtype=np.float32)
            e = [slice(None)] * ndim
            o = e.copy()
            e[axis] = slice(0, None, 2)
            o[axis] = slice(1, None, 2)
            out[tuple(e)] = (low+high) / np.sqrt(2.)
            out[tuple(o)] = (low-high) / np.sqrt(2.)
            result[prefix] = out
        current = result
    return current['']


def low_only(bands):
    return ihaar({name: value if set(name) == {'L'} else np.zeros_like(value)
                  for name, value in bands.items()})


def im(ax, a, title, kind='gray', limits=None, colorbar=False):
    a = np.asarray(a)
    if kind == 'label':
        obj = ax.imshow(a, cmap=LABEL_CMAP, norm=LABEL_NORM, interpolation='nearest')
    elif kind == 'signed':
        bound = max(float(np.max(np.abs(a))), 1e-8) if limits is None else limits
        obj = ax.imshow(a, cmap='RdBu_r', vmin=-bound, vmax=bound, interpolation='nearest')
    else:
        lo, hi = np.percentile(a, [1, 99]) if limits is None else limits
        if hi <= lo:
            hi = lo + 1e-6
        obj = ax.imshow(a, cmap='gray' if kind == 'gray' else kind,
                        vmin=lo, vmax=hi, interpolation='nearest')
    ax.set_title(title, fontsize=12)
    ax.axis('off')
    if colorbar:
        plt.colorbar(obj, ax=ax, fraction=.046, pad=.03)
    return obj


def finish(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def fft_parts(a, cutoff=.12):
    a = np.asarray(a, dtype=np.float32)
    fy, fx = np.meshgrid(np.fft.fftfreq(a.shape[0]), np.fft.fftfreq(a.shape[1]), indexing='ij')
    mask = np.hypot(fy, fx) <= cutoff
    spectrum = np.fft.fft2(a)
    low = np.fft.ifft2(spectrum * mask).real.astype(np.float32)
    high = a - low
    # Subtract the DC component for display only; reconstruction retains it.
    display = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(a-a.mean()))))
    return display, low, high


def frequency_panel(a, title, path, cutoff=.12):
    spectrum, low, high = fft_parts(a, cutoff)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    im(axes[0], a, title, colorbar=True)
    im(axes[1], spectrum, '去均值后的幅度谱\n中心=低频，四周=高频', 'magma')
    im(axes[2], low, f'低频重建 ≤ {cutoff} 周期/像素', colorbar=True)
    im(axes[3], high, '高频重建（有正负）', 'signed', colorbar=True)
    finish(fig, path)
    assert np.allclose(low+high, a, atol=1e-5)


def wavelet_levels(a, title, out, levels=3):
    """Only LL is sent to the next decomposition level, not all four bands."""
    current = np.asarray(a, dtype=np.float32)
    rows = []
    for level in range(1, levels+1):
        bands = haar(current)
        rec = ihaar(bands)
        approx = low_only(bands)
        scale = max(float(np.max(np.abs(v))) for k, v in bands.items() if k != 'LL')
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        im(axes[0,0], current, f'{title}：第{level}层输入\n{current.shape}', colorbar=True)
        im(axes[0,1], bands['LL'], f'LL：低频系数 {bands["LL"].shape}', colorbar=True)
        im(axes[0,2], bands['LH'], 'LH：y低通、x高通', 'signed', scale, True)
        im(axes[1,0], bands['HL'], 'HL：y高通、x低通', 'signed', scale, True)
        im(axes[1,1], bands['HH'], 'HH：y高通、x高通', 'signed', scale, True)
        im(axes[1,2], approx, '只保留LL后的重建\n与本层输入同尺寸', colorbar=True)
        finish(fig, out / f'{title}_haar_level{level}.png')
        energy = float(sum(np.sum(v.astype(np.float64)**2) for v in bands.values()))
        input_energy = float(np.sum(current.astype(np.float64)**2))
        error = float(np.max(np.abs(rec-current)))
        assert error < 2e-4 * max(1., float(np.max(np.abs(current))))
        assert np.isclose(energy, input_energy, rtol=1e-5, atol=1e-7)
        rows.append(dict(对象=title, 小波层=level, 输入尺寸=str(current.shape),
                         子带尺寸=str(bands['LL'].shape), 完整重建最大误差=error,
                         细节能量比例=sum(float(np.sum(v.astype(np.float64)**2)) for k,v in bands.items() if k!='LL')/max(energy,1e-20)))
        current = bands['LL']
    return rows


def load_case(data_root, case, patch_size):
    import blosc2
    blosc2.set_nthreads(1)
    folder = Path(data_root) / 'nnUNetPlans_3d_fullres'
    data = blosc2.open(urlpath=str(folder / f'{case}.b2nd'), mode='r', dparams={'nthreads': 1})
    seg_arr = blosc2.open(urlpath=str(folder / f'{case}_seg.b2nd'), mode='r', dparams={'nthreads': 1})
    seg = np.asarray(seg_arr[:])[0]
    counts = (seg == 2).sum(axis=(1, 2))
    if counts.max() == 0:
        raise ValueError('Selected case contains no tumor; choose another validation case')
    z = int(counts.argmax())
    points = np.argwhere(seg[z] == 2)
    # A real tumor pixel nearest the 2D centroid avoids placing the center in a gap.
    center_yx = points[np.argmin(((points-points.mean(0))**2).sum(1))]
    center = np.array([z, *center_yx], dtype=int)
    starts = center - patch_size//2
    ends = starts + patch_size
    lo = np.maximum(starts, 0)
    hi = np.minimum(ends, seg.shape)
    slices = tuple(slice(int(a), int(b)) for a,b in zip(lo, hi))
    pad = tuple((int(a),int(b)) for a,b in zip(lo-starts, ends-hi))
    image = np.pad(np.asarray(data[(0,)+slices]), pad, mode='constant')
    label = np.pad(seg[slices], pad, mode='constant', constant_values=0)
    if np.any(label < 0):
        raise ValueError('Ignore labels exist in this crop; choose another crop or explicitly mask them')
    assert image.shape == label.shape == (patch_size,)*3
    record = dict(case=case, subset='LiTS validation', full_preprocessed_shape=list(seg.shape),
                  tumor_center_zyx=center.tolist(), crop_start_zyx=starts.tolist(), padding=pad,
                  patch_shape=list(image.shape), data_path=str(folder/f'{case}.b2nd'),
                  label_path=str(folder/f'{case}_seg.b2nd'), selection='GT-selected tumor-bearing patch; exploratory only')
    return image.astype(np.float32), label.astype(np.int16), np.asarray(data[0,z,:,:]), seg[z], record


def capture_model(image, label, source, out, threads=4):
    """One frozen CPU forward. Hooks retain all channels of one 2D plane only."""
    import torch
    from pumengyu.architectures.mednext import build_mednext_large_mha
    torch.set_num_threads(threads)
    state = torch.load(source, map_location='cpu', weights_only=False)
    model = build_mednext_large_mha(1, 3, enable_deep_supervision=False, mha_use_moe=False)
    model.load_state_dict(state['network_weights'], strict=True)
    model.outside_block_checkpointing = False  # inference only: identical layers, no backward recomputation
    model.eval().requires_grad_(False)
    epoch, trainer = state['current_epoch'], state['trainer_name']
    del state
    names = ['stem']
    for i in range(4):
        names += [f'enc_block_{i}', f'down_{i}']
    names += ['bottleneck', 'mla_bot']  # historical attribute name; instance is MHABottleneck3D
    for i in (3,2,1,0):
        names += [f'up_{i}', f'dec_block_{i}']
    names += ['out_0']
    arrays, rows, handles = {}, [], []

    def hook(name):
        def save(module, inputs, result):
            before = inputs[0].detach()
            after = result.detach()
            row = dict(module=name, type=type(module).__name__,
                       input_shape=list(before.shape), output_shape=list(after.shape))
            for which, tensor in [('in', before), ('out', after)]:
                z = min(tensor.shape[2]-1, int((image.shape[0]//2+.5)*tensor.shape[2]/image.shape[0]))
                plane = tensor[0,:,z].float().cpu().numpy().copy()
                arrays[f'{name}__{which}'] = plane
                row[f'{which}_z_index'] = z
                row[f'{which}_plane_min'] = float(plane.min())
                row[f'{which}_plane_max'] = float(plane.max())
            rows.append(row)
        return save

    for name in names:
        handles.append(getattr(model, name).register_forward_hook(hook(name)))
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            logits = model(torch.from_numpy(image[None,None]).float())
        assert logits.shape == (1,3,*image.shape)
        prediction = logits.argmax(1)[0].cpu().numpy().astype(np.int16)
        probability = logits.softmax(1)[0,2].cpu().numpy()
    finally:
        for handle in handles:
            handle.remove()
    np.savez_compressed(out/'stage_planes.npz', **arrays)
    np.savez_compressed(out/'patch_prediction.npz', prediction=prediction, tumor_probability=probability)
    metadata = dict(checkpoint=str(source), checkpoint_sha256=sha(source), epoch=epoch, trainer=trainer,
                    model='MedNeXt-L + MHA; no MoE; no frequency or wavelet branch',
                    source_code='pumengyu/architectures/mednext.py', device='cpu', torch_threads=threads,
                    eval_mode=True, patch_shape=list(image.shape), elapsed_seconds=time.perf_counter()-started,
                    inference='single GT-centered patch, no TTA, no sliding window; not formal evaluation',
                    rows=rows)
    (out/'stage_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    del model, logits
    return arrays, metadata, prediction, probability


def resize_label_views(label, size):
    """Visualization only: hard nearest label and soft tumor occupancy on a 2D plane."""
    import torch
    from torch.nn.functional import interpolate
    y = torch.from_numpy(label.astype(np.float32))[None,None]
    hard = interpolate(y, size=size, mode='nearest')[0,0].numpy()
    coverage = interpolate((y==2).float(), size=size, mode='area')[0,0].numpy()
    return hard, coverage


def stage_panel(name, arrays, label, out, channel=0):
    a, b = arrays[name+'__in'], arrays[name+'__out']
    ci, co = min(channel,a.shape[0]-1), min(channel,b.shape[0]-1)
    hard, coverage = resize_label_views(label, b.shape[-2:])
    fig, axes = plt.subplots(2,2,figsize=(10,8))
    im(axes[0,0], a[ci], f'{name} 输入 ch{ci}\n切片 C×H×W={a.shape}', 'signed', colorbar=True)
    im(axes[0,1], b[co], f'{name} 输出 ch{co}\n切片 C×H×W={b.shape}', 'signed', colorbar=True)
    im(axes[1,0], np.sqrt(np.mean(b*b, axis=0)), '输出：通道RMS\n不是概率或分割图', 'magma', colorbar=True)
    im(axes[1,1], coverage, '原切片肿瘤面积覆盖率\n仅作位置参照；不输入模型', 'magma', (0,1), True)
    finish(fig, out/f'{name}_input_output.png')
    return dict(stage=name, 输出网格=str(b.shape[-2:]), 最近邻标签肿瘤像素=int((hard==2).sum()),
                覆盖率最大值=float(coverage.max()),覆盖率非零格数=int((coverage>0).sum()))

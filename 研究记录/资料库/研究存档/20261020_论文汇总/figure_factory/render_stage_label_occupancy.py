"""Compare native feature grids to geometric label occupancy, without contours."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np


def pool_occupancy(label, shape):
    """Disjoint input-grid bins, not convolution receptive fields."""
    factors = tuple(a // b for a, b in zip(label.shape, shape))
    assert all(a == b * f for a, b, f in zip(label.shape, shape, factors))
    view = (shape[0], factors[0], shape[1], factors[1], shape[2], factors[2])
    tumor = (label == 2).reshape(view).sum(axis=(1, 3, 5))
    valid = (label >= 0).reshape(view).sum(axis=(1, 3, 5))
    occupancy = np.divide(tumor, valid, out=np.full(shape, np.nan, dtype=np.float64), where=valid > 0)
    return occupancy, valid, factors


def main():
    root = Path(__file__).resolve().parents[1] / '方案1_问题定义与思路检索/skip诊断/liver_93_stage1_20261014'
    meta = json.loads((root / 'summary.json').read_text())
    plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    outputs, record = {}, {}
    with np.load(root / 'patch_predictions.npz') as data, np.load(root / 'feature_slices.npz') as features:
        label = data['label']
        for kind, order, role in [('encoder', range(4), 'encoder'), ('decoder', (3, 2, 1, 0), 'decoded')]:
            panels = []
            for i in order:
                key = f'skip_{i}.{role}'
                shape = tuple(meta['feature_stats'][key]['shape'][-3:])
                zi = meta['feature_stats'][key]['native_slice']
                occ, valid, factors = pool_occupancy(label, shape)
                enlarged = features[key][0]
                # Saved nearest-neighbor enlargement repeats each native value
                # exactly. Recover and verify it, rather than smoothing a map.
                native = enlarged[::factors[1], ::factors[2]]
                assert native.shape == shape[1:]
                reconstructed = np.repeat(np.repeat(native, factors[1], axis=0), factors[2], axis=1)
                np.testing.assert_array_equal(reconstructed, enlarged)
                name = f'{kind}_stage_{i+1}'
                outputs[name + '_feature_rms'] = native
                outputs[name + '_tumor_occupancy'] = occ[zi]
                outputs[name + '_valid_count'] = valid[zi]
                assert np.isclose(np.nansum(occ * valid), (label == 2).sum())
                panels.append((native, occ[zi]))
                record[name] = dict(native_shape=shape, native_slice=zi, geometric_bin=factors,
                                    input_depth_range_inclusive=[zi * factors[0], (zi + 1) * factors[0] - 1])
            vmax = max(float(np.percentile(a, 99)) for a, _ in panels) or 1
            fig = plt.figure(figsize=(9.4, 17.2))
            cmap_occ = plt.get_cmap('viridis').copy(); cmap_occ.set_bad('#bbbbbb')
            for row, (native, occupancy) in enumerate(panels):
                for col, arr in enumerate((native, occupancy)):
                    ax = fig.add_axes([.008 + col * .505, .06 + (3 - row) * .232, .48, .23])
                    ax.imshow(arr, interpolation='nearest', cmap='magma' if col == 0 else cmap_occ,
                              vmin=0, vmax=vmax if col == 0 else 1)
                    ax.text(.02, .98, chr(97 + row * 2 + col), transform=ax.transAxes, va='top',
                            fontsize=24, weight='bold', bbox=dict(facecolor='white', alpha=.9, edgecolor='none', pad=2))
                    ax.axis('off')
            # Different units require different scales, each shared by its column.
            for col, (norm, cmap, title) in enumerate([(Normalize(0, vmax), 'magma', '特征响应强度'),
                                                       (Normalize(0, 1), cmap_occ, '标注肿瘤占比')]):
                cax = fig.add_axes([.03 + col * .505, .035, .43, .012])
                cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal')
                cb.ax.tick_params(labelsize=13, pad=2)
                cb.ax.set_title(title, fontsize=14, pad=6)
            fig.savefig(root / f'test_viz/08_{kind}_native_occupancy.png', dpi=300, bbox_inches='tight', pad_inches=.02)
            plt.close(fig)
    np.savez_compressed(root / 'native_grid_occupancy.npz', **outputs)
    record['definition'] = 'Number of tumor-labelled input voxels / valid input voxels within disjoint aligned grid bins. Not feature receptive-field occupancy or model prediction.'
    (root / 'native_grid_occupancy.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print('Native grids recovered exactly; occupancy counts conserved; encoder and decoder comparisons exported.')


if __name__ == '__main__':
    main()

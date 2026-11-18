"""Render the unmodified network's final tumor prediction and error ROIs."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
from scipy import ndimage


def main():
    root = Path(__file__).resolve().parents[1] / '方案1_问题定义与思路检索/skip诊断/liver_93_stage1_20261014'
    meta = json.loads((root / 'summary.json').read_text())
    z = meta['slice_z']
    plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
    with np.load(root / 'patch_predictions.npz') as data:
        ct, gt, pred = data['image'][0, z], data['label'][z] == 2, data['scale_1'][z] == 2
        valid = data['label'][z] >= 0
        fn, fp = gt & ~pred & valid, ~gt & pred & valid
        tp = gt & pred & valid
        rgba = np.zeros((*gt.shape, 4))
        rgba[fn] = [0, .6, 1, .9]; rgba[fp] = [1, 0, 0, .9]
        lo, hi = np.percentile(ct, [1, 99])
        rois, records = [], {}
        for name, mask in [('漏检', fn), ('误报', fp)]:
            ids, n = ndimage.label(mask, np.ones((3, 3)))
            sizes = np.bincount(ids.ravel()); sizes[0] = 0
            candidates = np.argsort(sizes)[::-1]
            selected = None
            for ident in candidates:
                if ident == 0 or sizes[ident] == 0: continue
                coords = np.argwhere(ids == ident)
                if not ((coords == 0).any() or (coords[:, 0] == gt.shape[0]-1).any() or (coords[:, 1] == gt.shape[1]-1).any()):
                    selected = int(ident); break
            if selected is None: selected = int(sizes.argmax())
            center = np.asarray(ndimage.center_of_mass(mask, ids, selected)) if n else np.asarray(gt.shape)/2
            size = 40
            y, x = np.clip(np.rint(center).astype(int)-size//2, 0, np.asarray(gt.shape)-size)
            rois.append((name, int(y), int(x), size))
            records[name] = dict(y=int(y), x=int(x), size=size, selected_component_pixels=int(sizes[selected]),
                                 selection='largest error component not touching patch edge, if available')

        def panel(ax, letter, region=None, show_gt=False, show_pred=False, errors=False):
            sl = region or (slice(None), slice(None))
            ax.imshow(ct[sl], cmap='gray', vmin=lo, vmax=hi, interpolation='nearest')
            if errors: ax.imshow(rgba[sl], interpolation='nearest')
            for mask, color, enabled in [(gt, '#ffdd00', show_gt), (pred, '#00ffff', show_pred)]:
                if enabled and mask[sl].any() and not mask[sl].all():
                    ax.contour(mask[sl], levels=[.5], colors=[color], linewidths=1.0)
            ax.text(.02, .98, letter, transform=ax.transAxes, va='top', fontsize=25, weight='bold',
                    bbox=dict(facecolor='white', alpha=.9, edgecolor='none', pad=2))
            ax.axis('off')

        fig, axes = plt.subplots(2, 2, figsize=(9, 9.4))
        panel(axes[0, 0], 'a')
        panel(axes[0, 1], 'b', show_gt=True)
        panel(axes[1, 0], 'c', show_pred=True)
        panel(axes[1, 1], 'd', errors=True)
        for i, (_, y, x, size) in enumerate(rois):
            axes[1, 1].add_patch(Rectangle((x-.5, y-.5), size, size, fill=False,
                                         edgecolor='white', linewidth=1.3, linestyle='--'))
            axes[1, 1].text(x, y, str(i+1), color='white', fontsize=14, weight='bold')
        legend = [Patch(color='#ffdd00', label='标注肿瘤轮廓'), Patch(color='#00ffff', label='预测肿瘤轮廓'),
                  Patch(color='#0099ff', label='漏检'), Patch(color='red', label='误报')]
        fig.legend(handles=legend, loc='lower center', ncol=4, frameon=False, fontsize=12)
        fig.subplots_adjust(left=.005, right=.995, top=.995, bottom=.065, wspace=.025, hspace=.025)
        fig.savefig(root/'test_viz/09_final_prediction_errors.png', dpi=300, bbox_inches='tight', pad_inches=.02)
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(9, 9.4))
        for row, (_, y, x, size) in enumerate(rois):
            sl = (slice(y, y+size), slice(x, x+size))
            panel(axes[row, 0], chr(97+row*2), sl, show_gt=True, show_pred=True)
            panel(axes[row, 1], chr(98+row*2), sl, errors=True)
        fig.legend(handles=legend, loc='lower center', ncol=4, frameon=False, fontsize=12)
        fig.subplots_adjust(left=.005, right=.995, top=.995, bottom=.065, wspace=.025, hspace=.025)
        fig.savefig(root/'test_viz/09_final_prediction_error_zoom.png', dpi=300, bbox_inches='tight', pad_inches=.02)
        plt.close(fig)
        report = dict(slice_z=z, prediction='scale_1: original network, skip unchanged',
                      slice_pixels=dict(tumor_gt=int(gt.sum()),tp=int(tp.sum()),fp=int(fp.sum()),fn=int(fn.sum())),
                      rois=records, note='Single patch, not whole-case inference; patch-edge errors require full-case confirmation.')
        (root/'final_prediction_errors.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()

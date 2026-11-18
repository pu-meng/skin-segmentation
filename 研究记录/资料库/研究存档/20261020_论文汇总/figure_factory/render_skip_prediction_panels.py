"""Render large, individually labelled panels from saved diagnostic arrays."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import numpy as np
from scipy import ndimage


def render_prediction_changes(root):
    """Show only changed tumor decisions, on the same original slice and ROI."""
    z = json.loads((root / 'summary.json').read_text())['slice_z']
    with np.load(root / 'patch_predictions.npz') as data:
        valid, truth = data['label'] >= 0, data['label'] == 2
        baseline = data['scale_1'] == 2
        ct = data['image'][0, z]
        records, changes = {}, {}
        for key in ('scale_0.5', 'scale_0'):
            pred = data[key] == 2
            fixed = (baseline != truth) & (pred == truth) & valid
            harmed = (baseline == truth) & (pred != truth) & valid
            assert np.array_equal(fixed | harmed, (pred != baseline) & valid)
            changes[key] = (fixed[z], harmed[z])
            records[key] = {
                'whole_patch': dict(corrected=int(fixed.sum()), damaged=int(harmed.sum())),
                'display_slice': dict(corrected=int(fixed[z].sum()), damaged=int(harmed[z].sum())),
            }
        union = np.logical_or.reduce([mask for pair in changes.values() for mask in pair])
        components, n = ndimage.label(union, structure=np.ones((3, 3)))
        if n:
            counts = np.bincount(components.ravel()); counts[0] = 0
            center = np.mean(np.argwhere(components == counts.argmax()), axis=0)
        else:
            center = np.asarray(ct.shape) / 2
        size = 40
        start = np.clip(np.rint(center).astype(int) - size // 2, 0, np.asarray(ct.shape) - size)
        y0, x0 = map(int, start)
        limits = np.percentile(ct, [1, 99])
        fig, axes = plt.subplots(2, 2, figsize=(9, 9.3))
        for col, key in enumerate(('scale_0.5', 'scale_0')):
            fixed, harmed = changes[key]
            rgba = np.zeros((*ct.shape, 4), dtype=np.float32)
            rgba[fixed] = [0, 1, .35, 1]
            rgba[harmed] = [1, .2, .75, 1]
            for row in range(2):
                ax = axes[row, col]
                region = (slice(None), slice(None)) if row == 0 else (slice(y0, y0 + size), slice(x0, x0 + size))
                ax.imshow(ct[region], cmap='gray', vmin=limits[0], vmax=limits[1], interpolation='nearest')
                ax.imshow(rgba[region], interpolation='nearest')
                if row == 0:
                    ax.add_patch(plt.Rectangle((x0 - .5, y0 - .5), size, size, fill=False,
                                              edgecolor='#ffde59', linewidth=1.4))
                ax.text(.02, .98, chr(97 + row * 2 + col), transform=ax.transAxes, va='top',
                        fontsize=25, weight='bold', bbox=dict(facecolor='white', alpha=.9, edgecolor='none', pad=2))
                ax.axis('off')
        fig.legend(handles=[Patch(color='#00ff59', label='原来错 → 改对'),
                            Patch(color='#ff33bf', label='原来对 → 改错'),
                            Patch(facecolor='none', edgecolor='#ffde59', label='下排放大区域')],
                   loc='lower center', ncol=3, frameon=False, fontsize=13)
        fig.subplots_adjust(left=.005, right=.995, top=.995, bottom=.065, wspace=.025, hspace=.025)
        fig.savefig(root / 'test_viz/00_prediction_changes.png', dpi=300, bbox_inches='tight', pad_inches=.02)
        plt.close(fig)
        records['slice_z'] = z
        records['zoom_box_yx_size'] = [y0, x0, size]
        records['zoom_selection'] = 'Largest connected component of changed tumor decisions on original display slice; same ROI for both interventions.'
        records['definition'] = 'Binary tumor versus non-tumor correctness; whole_patch counts use all valid 128^3 patch voxels.'
        (root / 'tumor_change_summary.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')


def render_features(root):
    """Two large panels per row, compact margins, one raw-value colorbar."""
    with np.load(root / 'feature_slices.npz') as saved, np.load(root / 'patch_predictions.npz') as data:
        z = json.loads((root / 'summary.json').read_text())['slice_z']
        tumor = data['label'][z] == 2

        def draw(keys, filename, channel=0, signed=False):
            arrays = [saved[key][channel] for key in keys]
            vals = np.concatenate([arr.ravel() for arr in arrays])
            vmax = float(np.percentile(np.abs(vals) if signed else vals, 99)) or 1.
            rows = (len(arrays) + 1) // 2
            single = len(arrays) == 1
            fig = plt.figure(figsize=(4.8 if single else 8.6, 4.1 * rows))
            for i, arr in enumerate(arrays):
                row, col = divmod(i, 2)
                # No external per-panel title/colorbar: image area fills cell.
                rect = [.012, .02, .83, .96] if single else [
                    .012 + col * .449, 1 - (row + 1) / rows + .012 / rows, .441, .976 / rows]
                ax = fig.add_axes(rect)
                im = ax.imshow(arr, cmap='RdBu_r' if signed else 'magma',
                               vmin=-vmax if signed else 0, vmax=vmax, interpolation='nearest')
                if tumor.any() and not tumor.all():
                    ax.contour(tumor, levels=[.5], colors=['lime'], linewidths=.8)
                ax.text(.025, .975, chr(97 + i), transform=ax.transAxes,
                        va='top', fontsize=24, weight='bold', color='black',
                        bbox=dict(facecolor='white', edgecolor='none', alpha=.9, pad=2))
                ax.axis('off')
            bar = fig.add_axes([.87, .12, .035, .76] if single else [.929, .12, .022, .76])
            cb = fig.colorbar(im, cax=bar)
            cb.ax.tick_params(labelsize=15, length=3, pad=4)
            cb.locator = matplotlib.ticker.MaxNLocator(nbins=5)
            cb.update_ticks()
            fig.savefig(root / 'test_viz' / filename, dpi=300, bbox_inches='tight', pad_inches=.02)
            plt.close(fig)

        draw(['stage_input'] + [f'block_{i}.output' for i in range(4)] + ['down_output'], '01_stage_blocks.png')
        for i in range(4):
            draw([f'block_{i}.{op}' for op in ('conv1', 'norm', 'conv2', 'act', 'conv3', 'output')],
                 f'02_block_{i}_operators.png')
        for i in range(4):
            draw([f'skip_{i}.{op}' for op in ('encoder', 'up', 'sum', 'decoded')], f'03_skip_{i}_path.png')
        draw([f'block_{i}.output' for i in range(4)], '04_fixed_channel_0.png', channel=1, signed=True)
        draw([f'skip_{i}.encoder' for i in range(4)], '05_encoder_stage_outputs.png')
        draw([f'skip_{i}.decoded' for i in (3, 2, 1, 0)], '06_decoder_stage_outputs.png')
        draw([f'skip_{i}.{op}' for i in range(4) for op in ('encoder', 'decoded')],
             '07_encoder_decoder_stage_pairs.png')
        for i in range(4):
            draw([f'skip_{i}.encoder'], f'encoder_stage_{i+1}_output.png')
            draw([f'skip_{i}.decoded'], f'decoder_stage_{i+1}_output.png')


def render_prediction_grid(root):
    z = json.loads((root / 'summary.json').read_text())['slice_z']
    with np.load(root / 'patch_predictions.npz') as data:
        ct, gt = data['image'][0, z], data['label'][z]
        limits = np.percentile(ct, [1, 99])
        fig = plt.figure(figsize=(8.6, 12.3))
        for i, key in enumerate([None, 'gt', 'scale_1', 'scale_0.5', 'scale_0']):
            row, col = divmod(i, 2)
            ax = fig.add_axes([.012 + col * .494, 1 - (row + 1) / 3 + .012 / 3, .484, .976 / 3])
            if key == 'gt':
                ax.imshow(gt, cmap=ListedColormap(['#292929', '#52b6b0', '#ffd841']),
                          norm=BoundaryNorm([-.5, .5, 1.5, 2.5], 3), interpolation='nearest')
            else:
                ax.imshow(ct, cmap='gray', vmin=limits[0], vmax=limits[1], interpolation='nearest')
                if key:
                    pred = data[key][z]
                    error = np.zeros((*gt.shape, 4), dtype=np.float32)
                    error[(pred == 2) & (gt == 2)] = [0, 1, 0, .8]
                    error[(pred == 2) & (gt != 2) & (gt >= 0)] = [1, 0, 0, .8]
                    error[(pred != 2) & (gt == 2)] = [0, .6, 1, .8]
                    ax.imshow(error, interpolation='nearest')
            ax.text(.025, .975, chr(97 + i), transform=ax.transAxes, va='top', fontsize=24,
                    weight='bold', bbox=dict(facecolor='white', edgecolor='none', alpha=.9, pad=2))
            ax.axis('off')
        ax = fig.add_axes([.52, .02, .47, .29]); ax.axis('off')
        ax.text(.02, .93, '人工标注（b）', fontsize=18, weight='bold', transform=ax.transAxes)
        for j, (color, text) in enumerate([('#292929', '背景'), ('#52b6b0', '肝脏'), ('#ffd841', '肿瘤')]):
            ax.add_patch(plt.Rectangle((.04, .77 - j * .11), .07, .055, transform=ax.transAxes, color=color))
            ax.text(.15, .77 - j * .11, text, fontsize=16, transform=ax.transAxes)
        ax.text(.02, .43, '预测误差（c、d、e）', fontsize=18, weight='bold', transform=ax.transAxes)
        for j, (color, text) in enumerate([('#00cc00', '肿瘤预测正确'), ('red', '误报'), ('#0099ff', '漏检')]):
            ax.add_patch(plt.Rectangle((.04, .27 - j * .11), .07, .055, transform=ax.transAxes, color=color))
            ax.text(.15, .27 - j * .11, text, fontsize=16, transform=ax.transAxes)
        fig.savefig(root / 'test_viz/00_prediction_grid.png', dpi=300, bbox_inches='tight', pad_inches=.02)
        plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[1] / '方案1_问题定义与思路检索/skip诊断/liver_93_stage1_20261014'
    meta = json.loads((root / 'summary.json').read_text())
    z = meta['slice_z']
    plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    with np.load(root / 'patch_predictions.npz') as data:
        ct, gt = data['image'][0, z], data['label'][z]
        limits = np.percentile(ct, [1, 99])
        specs = [
            ('00a_input_ct.png', '① 输入模型的CT切片', None, None),
            ('00b_manual_label.png', '② 人工标注：用于对照模型预测', None, None),
            ('00c_original_prediction.png', '③ 原模型预测：完整传递编码特征', 'scale_1', '编码特征 ＋ 上采样特征 → 继续解码 → 分割'),
            ('00d_half_skip_prediction.png', '④ 将skip 1传递的编码特征乘以0.5', 'scale_0.5', '0.5 × 编码特征 ＋ 上采样特征 → 继续解码 → 分割'),
            ('00e_zero_skip_prediction.png', '⑤ 不通过skip 1传递编码特征', 'scale_0', '只保留该处上采样特征 → 继续解码 → 分割'),
        ]
        for name, title, key, subtitle in specs:
            fig, ax = plt.subplots(figsize=(7, 7.7))
            if name == '00b_manual_label.png':
                cmap = ListedColormap(['#292929', '#52b6b0', '#ffd841'])
                ax.imshow(gt, cmap=cmap, norm=BoundaryNorm([-.5, .5, 1.5, 2.5], 3), interpolation='nearest')
                legend = [Patch(color=c, label=l) for c, l in zip(cmap.colors, ['背景', '肝脏', '肿瘤'])]
            else:
                ax.imshow(ct, cmap='gray', vmin=limits[0], vmax=limits[1], interpolation='nearest')
                legend = []
                if key:
                    pred = data[key][z]
                    error = np.zeros((*gt.shape, 4), dtype=np.float32)
                    error[(pred == 2) & (gt == 2)] = [0, 1, 0, .8]
                    error[(pred == 2) & (gt != 2) & (gt >= 0)] = [1, 0, 0, .8]
                    error[(pred != 2) & (gt == 2)] = [0, .6, 1, .8]
                    ax.imshow(error, interpolation='nearest')
                    legend = [Patch(color=c, label=l) for c, l in [('#00cc00', '肿瘤预测正确'), ('red', '误报'), ('#0099ff', '漏检')]]
            ax.set_title(title, fontsize=15, pad=14, weight='bold')
            ax.axis('off')
            if legend:
                ax.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, -.02), ncol=3, frameon=False, fontsize=11)
            fig.text(.5, .025, subtitle or 'MedNeXt-L＋MHA＋MoE · liver_93 · 局部切片76', ha='center', fontsize=11)
            fig.subplots_adjust(top=.9, bottom=.13, left=.03, right=.97)
            fig.savefig(root / 'test_viz' / name, dpi=300)
            plt.close(fig)
    render_features(root)
    render_prediction_grid(root)
    render_prediction_changes(root)


if __name__ == '__main__':
    main()

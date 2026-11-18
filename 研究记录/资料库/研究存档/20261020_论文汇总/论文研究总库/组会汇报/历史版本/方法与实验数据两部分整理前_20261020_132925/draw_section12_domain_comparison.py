"""Plot the six adapted models in meeting-note sections 1 and 2, without pooling runs."""
from pathlib import Path
import hashlib
import json
import re
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPORT = Path('/home/PuMengYu/论文汇总/论文研究总库/组会汇报/近期研究进展与实验结论.md')
DEST = REPORT.parent / 'figures'
BLUE, ORANGE = '#3975a4', '#c47732'
TEXT = '#26394a'


def source_rows(metric='肿瘤Dice'):
    text = REPORT.read_text()
    first = text[text.index('## 1. LiTS-only'):text.index('<a id="hcc适配"></a>')]
    second = text[text.index('## 2. HCC三例适配'):text.index('### 2.2')]
    result = []
    names = [('A05', '无融合'), ('C13', '全局比例'), ('C14', '逐位置比例'),
             ('C0', '全参数'), ('C1', '固定编码器'), ('C2', 'LoRA')]
    for group, section in enumerate([first, second], 1):
        block = section.split(f'**完整CT测试：{metric}**', 1)[1]
        rows = []
        for line in block.splitlines():
            if not line.startswith('|'):
                if rows:
                    break
                continue
            cells = [s.strip() for s in line.strip('|').split('|')]
            if len(cells) == 4 and re.match(r'^\d', cells[1]):
                rows.append(cells)
        assert len(rows) == 4, 'Expected source plus three modified models'
        for cells, (code, label) in zip(rows[1:], names[(group-1)*3:group*3]):
            assert code in cells[0]
            values = [float(re.sub('[↑↓＝]', '', value)) for value in cells[1:]]
            result.append(dict(section=group, code=code, label=label,
                               scores=dict(zip(['LiTS', 'IRCADb', 'HCC'], values))))
    return result


def draw(metric):
    rows = source_rows(metric)
    stem = f'第1与第2节_三域{metric}对比'
    plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'svg.fonttype': 'none',
                         'axes.unicode_minus': False, 'font.size': 14,
                         'text.color': TEXT, 'axes.labelcolor': TEXT,
                         'xtick.color': TEXT, 'ytick.color': TEXT})
    fig, axes = plt.subplots(3, 1, figsize=(9, 12.2))
    fig.subplots_adjust(left=.245, right=.94, top=.82, bottom=.065, hspace=.47)
    fig.set_facecolor('white')
    title = ('第1节模型：LiTS/IRCADb较高，HCC较低' if metric == '肿瘤Dice'
             else '综合Dice：LiTS/IRCADb较高，HCC有交叠')
    fig.suptitle(title,
                 x=.5, y=.984, fontsize=19, fontweight='bold')
    fig.text(.5, .95, f'{metric}；每根柱表示一个模型的完整CT测试结果',
             ha='center', va='center', fontsize=14)
    fig.legend(handles=[Patch(facecolor=BLUE, label='第1节：LiTS-only，无MoE源模型'),
                        Patch(facecolor=ORANGE, label='第2节：HCC三例适配，含MoE源模型')],
               loc='upper center', bbox_to_anchor=(.5, .932), frameon=False,
               ncol=1, fontsize=14, handlelength=1.3, labelspacing=.4)
    summaries = {}
    y = [0, 1, 2, 3.65, 4.65, 5.65]
    labels = [f"{r['code']}  {r['label']}" for r in rows]
    for ax, domain, n in zip(axes, ['LiTS', 'IRCADb', 'HCC'], [26, 20, 21]):
        values = [r['scores'][domain] for r in rows]
        first, second = values[:3], values[3:]
        if domain == 'LiTS':
            assert min(first) > max(second)
            direction = '第1节三组均高于第2节'
        elif domain == 'IRCADb':
            assert sum(a > b for a in first for b in second) == 8
            direction = '第1节整体较高，但有交叠'
        elif metric == '肿瘤Dice':
            assert max(first) < min(second)
            direction = '第2节三组均高于第1节'
        else:
            assert sum(b > a for a in first for b in second) == 8
            direction = '第2节整体较高，但有交叠'
        summaries[domain] = dict(section1_range=[min(first), max(first)],
                                 section2_range=[min(second), max(second)],
                                 observation=direction, n_test=n)
        ax.axhspan(-.6, 2.6, color=BLUE, alpha=.055, zorder=0)
        ax.axhspan(3.05, 6.25, color=ORANGE, alpha=.055, zorder=0)
        bars = ax.barh(y, values, height=.65, color=[BLUE]*3+[ORANGE]*3,
                       edgecolor='white', linewidth=.8, zorder=3)
        for bar, value in zip(bars, values):
            ax.text(value+1.3, bar.get_y()+bar.get_height()/2, f'{value:.2f}',
                    va='center', ha='left', fontsize=14, color=TEXT)
        ax.set_yticks(y, labels, fontsize=14)
        ax.tick_params(axis='y', length=0, pad=10)
        ax.set_ylim(6.3, -.7)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.tick_params(axis='x', labelsize=13, length=3, color='#8a99a4')
        ax.grid(axis='x', color='#d9e1e7', linewidth=.7, zorder=1)
        ax.set_axisbelow(True)
        ax.set_title(f'{domain}（{n}例） · {direction}', loc='left',
                     fontsize=15, fontweight='bold', pad=12)
        for side in ['top', 'right', 'left']:
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color('#aebbc5')
    axes[-1].set_xlabel(f'{metric}（0—100，越高越好）', fontsize=14, labelpad=10)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure_box = fig.bbox
    for artist in fig.findobj(match=matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        box = artist.get_window_extent(renderer)
        assert box.x0 >= figure_box.x0-1 and box.x1 <= figure_box.x1+1, artist.get_text()
        assert box.y0 >= figure_box.y0-1 and box.y1 <= figure_box.y1+1, artist.get_text()
    DEST.mkdir(exist_ok=True)
    svg = DEST / f'{stem}.svg'
    fig.savefig(svg)
    ET.parse(svg)
    preview = Path('/tmp/section12_domain_comparison_720px.png' if metric == '肿瘤Dice'
                   else '/tmp/section12_domain_comparison_overall_720px.png')
    fig.savefig(preview, dpi=720/fig.get_figwidth())
    plt.close(fig)
    record = dict(source_markdown=str(REPORT), data_scope='Modified models only; original source-model reference rows excluded',
                  section1_training='92 LiTS training cases; 50 additional epochs; no HCC fitting',
                  section2_training='3 HCC training cases; 300 additional epochs; no LiTS replay',
                  metric=('Patient mean tumor Dice over GT tumor-positive test patients; percentage scale'
                          if metric == '肿瘤Dice' else
                          'Arithmetic mean of liver patient-mean Dice over all test cases and tumor patient-mean Dice over GT tumor-positive cases; percentage scale'),
                  rows=rows, domain_observations=summaries,
                  limitations='Different source architectures, training epochs, trainable scopes, and selection domains; observational comparison, no causal isolation',
                  authoring_script=str(Path(__file__).resolve()),
                  authoring_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  svg=str(svg), preview=str(preview), preview_width_px=720)
    (DEST / f'{stem}_数据.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    print(svg)
    print(preview)


if __name__ == '__main__':
    for metric in ['肿瘤Dice', '综合Dice']:
        draw(metric)

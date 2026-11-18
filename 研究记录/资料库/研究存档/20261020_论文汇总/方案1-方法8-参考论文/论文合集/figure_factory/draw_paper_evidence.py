"""Render evidence summaries from the adjacent research database (no new data).

SVGs are the deliverables; 100-dpi previews go to /tmp/paper_evidence_preview.
Re-run: python figure_factory/draw_paper_evidence.py
"""
from pathlib import Path
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures'
PREVIEW = Path('/tmp/paper_evidence_preview')
FONT = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'svg.fonttype': 'path', 'font.size': 13})
BLUE = ('#eaf2fc', '#3e6798')
PURPLE = ('#f1ecfb', '#7955a6')
GREEN = ('#eaf5ee', '#467c5b')
ORANGE = ('#fff2df', '#a56b26')

def wrap(s, size, width):
    """Wrap to measured physical width; retain English words as units."""
    from matplotlib.backends.backend_agg import RendererAgg
    renderer = RendererAgg(800, 800, 72)
    font = FONT.copy()
    font.set_size(size)
    lines = []
    for paragraph in s.split('\n'):
        line = ''
        for token in re.findall(r'[A-Za-z0-9_.%/＋+-]+|.', paragraph):
            trial = line + token
            if line and renderer.get_text_width_height_descent(trial, font, False)[0] > width * 72:
                lines.append(line.rstrip())
                line = token.lstrip()
            else:
                line = trial
        lines.append(line.rstrip())
    return '\n'.join(lines)


def label(ax, x, y, s, size=15, weight='normal', **kw):
    font = FONT.copy()
    font.set_weight(weight)
    return ax.text(x, y, s, fontproperties=font, fontsize=size,
                   color='#263342', linespacing=1.3, **kw)


def canvas(title, subtitle, height=7.0):
    fig = plt.figure(figsize=(8, height), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 8), ylim=(0, height))
    ax.axis('off')
    label(ax, .3, height-.25, wrap(title, 18, 7.4), 18, 'bold', va='top')
    subtitle_y = height-.25-.32*len(wrap(title, 18, 7.4).split('\n'))-.16
    label(ax, .3, subtitle_y, wrap(subtitle, 14.5, 7.4), 14.5, va='top')
    return fig, ax


def box(ax, cx, cy, title, desc, w=7.25, h=.9, palette=BLUE):
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                 boxstyle='round,pad=0.02,rounding_size=0.10',
                 facecolor=palette[0], edgecolor=palette[1], linewidth=1.3))
    title = wrap(title, 16.5, w-.34)
    desc = wrap(desc, 15, w-.34)
    th = len(title.split('\n')) * .28
    dh = len(desc.split('\n')) * .26
    top = cy + (th + dh + .08)/2
    label(ax, cx, top, title, 16.5, 'bold', ha='center', va='top')
    label(ax, cx, top-th-.08, desc, 15, ha='center', va='top')


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                 arrowstyle='-|>', mutation_scale=15, linewidth=1.4,
                 color='#586371', shrinkA=0, shrinkB=0))


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    PREVIEW.mkdir(exist_ok=True)
    fig.savefig(OUT / f'{name}.svg')
    fig.savefig(PREVIEW / f'{name}.png', dpi=100)
    plt.close(fig)


def flow(name, title, subtitle, steps, conclusion, limit):
    title = wrap(title, 18, 7.4)
    subtitle = wrap(subtitle, 14.5, 7.4)
    header = .25 + .32*len(title.split('\n')) + .16 + .26*len(subtitle.split('\n')) + .24
    hs = [.28*len(wrap(head, 16.5, 6.91).split('\n')) +
          .26*len(wrap(desc, 15, 6.91).split('\n')) + .32
          for head, desc, _ in steps]
    conclusion = wrap(conclusion, 15, 7.25)
    limit = wrap(limit, 14.5, 7.25)
    footer = .3 + .27*len(conclusion.split('\n')) + .16 + .26*len(limit.split('\n')) + .25
    height = header + sum(hs) + .28*(len(steps)-1) + footer
    fig, ax = canvas(title, subtitle, height)
    top = height-header
    for i, ((head, desc, palette), h) in enumerate(zip(steps, hs)):
        cy = top-h/2
        box(ax, 4, cy, head, desc, h=h, palette=palette)
        if i:
            arrow(ax, 4, top+.25, 4, top+.025)
        top -= h+.28
    y = top+.28-.30
    label(ax, .375, y, conclusion, 15, 'bold', va='top')
    y -= .27*len(conclusion.split('\n'))+.16
    label(ax, .375, y, limit, 14.5, va='top')
    save(fig, name)

def cor():
    # Qualitative only: the existing note has no numerical CEI table.
    flow('Evidence_01_CoR', '论文1｜低路由分数，是否就代表专家没用？',
         'CEI：比较虚拟删除专家前后的损失；固定模型参数，不重新训练。', [
         ('同一模型与校准数据', '正常计算，记录损失与专家的路由分数', BLUE),
         ('虚拟删除一个目标专家', '重新前向计算，记录删除后的损失', PURPLE),
         ('比较删除前后', '删除后损失 − 正常损失：增加越多，当前贡献越大', ORANGE),
         ('记录中的关键观察', '部分专家路由分数低，但删除后损失明显增加', GREEN),
         ], '支持：路由分数不能完全代表专家的当前贡献。',
         '未证明：这些专家一定储存长尾知识，或路由错误导致全部幻觉。')

def cove_bars():
    fig, ax = canvas('论文2｜组合、路由与贡献监督，哪些设置有收益？',
                     'NYUD-v2 · Table 3 · ΔE：相对各任务最佳单专家的综合增益', 7.4)
    names = ['平均融合', '拼接融合', '只加任务条件路由', '加组合，无任务条件路由',
             '路由＋组合', '路由＋组合＋贡献监督', '路由＋组合＋噪声', '完整：再加贡献监督与噪声']
    values = [3.83, 5.49, 6.13, 7.47, 8.12, 8.80, 8.15, 9.74]
    chart = fig.add_axes([.43, .23, .48, .59])
    chart.barh(range(8), values, color=['#b9cce3']*4+['#c7b8e1']*3+['#7bb893'], height=.58)
    chart.set_yticks(range(8), names, fontproperties=FONT, fontsize=14.5)
    chart.invert_yaxis()
    chart.set_xlim(0, 11)
    chart.set_xticks([0, 5, 10], ['0%', '+5%', '+10%'])
    chart.tick_params(axis='y', length=0, pad=10)
    chart.spines[['top', 'right', 'left']].set_visible(False)
    chart.spines['bottom'].set_color('#c4ccd3')
    for i, v in enumerate(values):
        chart.text(v+.13, i, f'+{v:.2f}%', va='center', fontsize=14.5, color='#263342')
    label(ax, .45, 1.26, '读图：简单融合也有效；完整设置在这组比较中最高。', 15, 'bold')
    label(ax, .45, .78, '边界：ΔE不是Dice，也不是各任务都增加相同百分点。', 14.5)
    label(ax, .45, .38, '消融支持组件收益；不能据此证明早期偏置妨碍学习。', 14.5)
    save(fig, 'Evidence_02a_COVE_results')

def cove_flow():
    flow('Evidence_02b_COVE_credit', '论文2｜怎样得到“当前候选贡献”的监督？',
         '六个候选＝三个原始专家特征＋三个两两组合特征；来源：Eq.9–12。', [
         ('固定当前候选特征与确定性路由权重', '正常预测，得到逐像素任务损失', BLUE),
         ('临时移除一个候选', '该候选权重置零，其余权重重新归一化', PURPLE),
         ('只重算任务预测头', '不重跑全部专家；比较移除后损失与正常损失', ORANGE),
         ('得到候选目标权重', '损失变化汇聚到图像网格，转成路由监督', GREEN),
         ], '含义：移除后损失增加越多，当前预测越依赖该候选。',
         '边界：当前系统的依赖，不等于候选固有能力或全部潜在价值。')

def epo_flow():
    flow('Evidence_03a_EPO_probe', '论文3｜不改专家，只换组合，能否改善当前预测？',
         '来源：Sec.3.1；只检查最终答案经核验正确的推理轨迹。', [
         ('固定模型、输入上下文与目标词元', '指定一个词元位置、一个MoE层', BLUE),
         ('构造有限候选', '取路由分数最高的32个专家，采样32组替代组合', PURPLE),
         ('逐组试换，保持调用数量不变', '每次只替换目标词元在指定层的组合，其余保持原样', ORANGE),
         ('原组合＋32组替代组合，共33组', '比较目标词元概率，并统计原组合的排名', BLUE),
         ('Qwen3最后层的低置信组', '原组最优的位置占0.8%；候选最优平均高20.4个百分点', GREEN),
         ], '支持：在已检查条件下，原路由仍有可利用的改进空间。',
         '未证明：全局最优或整题答对；每组同规模，总诊断开销仍增加。')

def epo_results():
    fig, ax = canvas('论文3｜更新路由器后，结果是否全面改善？',
                     '只更新最后一个MoE层的路由器；专家及其他层保持冻结。', 8.1)
    box(ax, 4, 6.48, '下游结果：小幅变化',
        'AIME / HMMT：多数报告K处，pass@K高于默认模型', h=1.05, palette=GREEN)
    label(ax, .375, 5.62, '固定标准轨迹：A 原路由器 → B 更新后的路由器', 15.5, 'bold')
    rows = [
        ('低置信组：原组合排名第1的比例', '0.6% → 1.9%', '提高'),
        ('低置信组：原组合进入前5的比例', '11.2% → 17.9%', '提高'),
        ('低置信组：目标词元平均概率', '37.5% → 35.0%', '降低'),
        ('低置信组：与候选最优的概率差', '19.2 → 21.9个百分点', '扩大'),
    ]
    for i, (head, value, change) in enumerate(rows):
        y = 4.91 - i*.86
        ax.add_patch(FancyBboxPatch((.375, y-.37), 7.25, .74,
                     boxstyle='round,pad=0.02,rounding_size=0.06',
                     facecolor=GREEN[0] if i<2 else ORANGE[0], edgecolor='none'))
        label(ax, .55, y+.16, head, 15.5, va='center')
        label(ax, .55, y-.16, value, 16, 'bold', va='center')
        label(ax, 7.4, y-.16, change, 15, va='center', ha='right')
    label(ax, .375, 1.58, '读图：排名改善，不代表目标词元概率也同步改善。', 15, 'bold', va='top')
    label(ax, .375, 1.12, '边界：A、B分组成员可能不同，不能当作同一词元\n子集的因果差值。', 14.5, va='top')
    label(ax, .375, .36, '来源：Fig.3、Table 5；未凭曲线估写下游提升数值。', 14.5, va='center')
    save(fig, 'Evidence_03b_EPO_results')

if __name__ == '__main__':
    cor()
    cove_bars()
    cove_flow()
    epo_flow()
    epo_results()
    print(f'Generated 5 SVGs in {OUT}; previews in {PREVIEW}')

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[1]
FONT = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['svg.fonttype'] = 'path'
fig, ax = plt.subplots(figsize=(12, 11))
fig.subplots_adjust(0, 0, 1, 1)
ax.set(xlim=(0, 13), ylim=(0, 12))
ax.axis('off')
ink = '#28384A'
colors = {'feature': '#E8F2FC', 'expert': '#EAE9F9', 'gate': '#FFF0D8', 'output': '#E3F3EB'}

def label(x, y, s, size=13, color=ink, weight='normal', ha='center'):
    ax.text(x, y, s, fontproperties=FONT, fontsize=size, color=color,
            ha=ha, va='center', weight=weight, linespacing=1.3)

def box(x, y, w, title, subtitle='', kind='feature', h=.8):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
        boxstyle='round,pad=0.025,rounding_size=0.10', linewidth=1.4,
        edgecolor=ink, facecolor=colors[kind], zorder=3))
    label(x, y+.14 if subtitle else y, title, 16, weight='bold')
    if subtitle:
        label(x, y-.17, subtitle, 12)

def line(points, dashed=False, color=ink):
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=color, lw=1.5, linestyle='--' if dashed else '-', zorder=1)

def arrow(a, b, dashed=False, color=ink):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=13,
        lw=1.5, color=color, linestyle='--' if dashed else '-',
        shrinkA=0, shrinkB=0, zorder=2))

def node(x, y, symbol):
    ax.add_patch(Circle((x, y), .22, facecolor='white', edgecolor=ink, lw=1.5, zorder=3))
    label(x, y, symbol, 19)

label(6.5, 11.55, '02｜双分支加权传递', 23, weight='bold')
label(6.5, 11.08, '演示设计：在一个 encoder–decoder 跳跃连接处，学习怎样传递特征', 13)
label(4.1, 10.45, '特征传递', 15, weight='bold')
label(10.4, 10.45, '权重从哪里来', 15, weight='bold')
line([(8.15, 3.0), (8.15, 10.2)], color='#DCE3EA')

box(4.15, 9.7, 3.2, '编码器 skip 特征 E', '[B, C, D, H, W]')
line([(4.15, 9.3), (4.15, 8.9), (2.6, 8.9)])
line([(4.15, 8.9), (5.7, 8.9)])
arrow((2.6, 8.9), (2.6, 8.35))
arrow((5.7, 8.9), (5.7, 8.35))
box(2.6, 7.95, 2.5, '分支 1：卷积', '1×1×1，C → C', 'expert')
box(5.7, 7.95, 2.5, '分支 2：卷积', '1×1×1，C → C', 'expert')
arrow((2.6, 7.55), (2.6, 6.87))
arrow((5.7, 7.55), (5.7, 6.87))
node(2.6, 6.65, '×')
node(5.7, 6.65, '×')
label(1.25, 6.65, 'w1', 16, '#A66B10')
label(7.05, 6.65, 'w2', 16, '#A66B10')
arrow((1.58, 6.65), (2.38, 6.65), True, '#A66B10')
arrow((6.72, 6.65), (5.92, 6.65), True, '#A66B10')
label(4.15, 7.22, '参数独立', 11, '#687586')
line([(2.6, 6.43), (2.6, 5.8)])
line([(5.7, 6.43), (5.7, 5.8)])
arrow((2.6, 5.8), (3.93, 5.8))
arrow((5.7, 5.8), (4.37, 5.8))
node(4.15, 5.8, '+')
arrow((4.15, 5.58), (4.15, 4.92))
node(4.15, 4.7, '+')
line([(2.55, 9.7), (.55, 9.7), (.55, 4.7)])
arrow((.55, 4.7), (3.93, 4.7))
label(1.73, 4.97, '原始 E 直接保留', 11)
arrow((4.15, 4.48), (4.15, 3.8))
box(4.15, 3.4, 3.7, '增强后的 skip 特征 E′', '形状与 E 相同', 'output')
arrow((4.15, 3.0), (4.15, 2.52))
node(4.15, 2.3, '+')
box(6.7, 2.3, 2.7, '解码器特征 U', '上采样并对齐到 E')
arrow((5.35, 2.3), (4.37, 2.3))
arrow((4.15, 2.08), (4.15, 1.55))
box(4.15, 1.15, 3.7, '送入当前级解码器', '继续计算分割特征', 'output')

box(9.4, 9.65, 1.5, 'E', '编码器特征')
box(11.45, 9.65, 1.5, 'U', '解码器特征')
arrow((9.4, 9.25), (9.4, 8.45))
arrow((11.45, 9.25), (11.45, 8.45))
box(10.425, 8.0, 3.8, '权重生成模块', '共同读取 E 与 U', 'gate', h=.9)
arrow((9.4, 7.55), (9.4, 6.75), True, '#A66B10')
arrow((11.45, 7.55), (11.45, 6.75), True, '#A66B10')
box(9.4, 6.35, 1.5, 'w1', '分支 1 权重', 'gate')
box(11.45, 6.35, 1.5, 'w2', '分支 2 权重', 'gate')
label(10.425, 5.45, '用于左图两个 × 节点\n虚线表示权重，实线表示特征', 12)
label(10.425, 4.18, '两个分支读取同一份 E\n没有预设肝脏／肿瘤分工', 12)
label(10.425, 2.87, '尚待确定\n权重粒度与归一化方式\n权重生成模块的具体结构', 12, '#687586')
label(6.5, .35, 'E′ = E + w1 × Conv1(E) + w2 × Conv2(E)    ｜    当前级解码器输入 = U + E′', 13)

out = ROOT / 'figures' / '02_双分支加权传递.svg'
fig.savefig(out, format='svg', facecolor='white')
fig.savefig('/tmp/idea02_preview.png', dpi=120, facecolor='white')
print(out)

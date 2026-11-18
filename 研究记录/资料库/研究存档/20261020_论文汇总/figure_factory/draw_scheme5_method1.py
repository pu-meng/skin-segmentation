"""Method 1 candidate: copied encoder provenance and prediction flow.

Only writes the two Scheme-5 method-1 SVG assets; PNGs are temporary QA.
Backbone order checked against pumengyu/architectures/mednext.py.
The dual encoder candidate is not implemented in the model code.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / '方案5_问题定义与思路检索' / 'figures'
FONT = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['svg.fonttype'] = 'none'
INK = '#28384a'
COLORS = {'source': '#fff0c0', 'copy': '#e1eefb', 'mix': '#eaddf4', 'out': '#dff1e6'}

def start(title, height):
    fig, ax = plt.subplots(figsize=(10, height))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0, 10), ylim=(height, 0))
    ax.axis('off')
    text(ax, 5, .38, title, 20, True)
    return fig, ax

def text(ax, x, y, value, size=14, bold=False):
    return ax.text(x, y, value, ha='center', va='center', color=INK,
                   fontproperties=FONT, fontsize=size,
                   weight='bold' if bold else 'normal', linespacing=1.4)

def box(ax, x, y, w, title, sub='', kind='source', h=.85):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
                 boxstyle='round,pad=0,rounding_size=.09',
                 facecolor=COLORS[kind], edgecolor=INK, linewidth=1.3))
    text(ax, x, y-.16 if sub else y, title, 16, True)
    if sub:
        text(ax, x, y+.19, sub, 13)

def arrow(ax, points, dashed=False):
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=INK, lw=1.4,
                linestyle='--' if dashed else '-')
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>',
                  mutation_scale=13, lw=1.4, color=INK,
                  linestyle='--' if dashed else '-', shrinkA=0, shrinkB=0))

def save(fig, ax, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for item in ax.texts:
        bounds = item.get_window_extent(renderer)
        if not fig.bbox.contains(bounds.x0, bounds.y0) or not fig.bbox.contains(bounds.x1, bounds.y1):
            raise RuntimeError('Text outside canvas: ' + item.get_text())
    target = OUT / (name + '.svg')
    fig.savefig(target, metadata={'Date': None}, facecolor='white')
    ET.parse(target)
    fig.savefig(Path('/tmp') / (name + '_review.png'), dpi=120, facecolor='white')
    plt.close(fig)
    print(target)

def overview():
    fig, ax = start('图9  同一编码器的冻结原件与微调副本', 12)
    text(ax, 5, .84, '候选设计，尚未实现或验证；两份编码器结构相同', 14)
    text(ax, 5, 1.4, '① 准备与微调：第二路从哪里来', 17, True)
    box(ax, 5, 2.15, 5.0, 'LiTS训练所得的源编码器', '复制时：结构和初始权重完全相同')
    arrow(ax, [(5, 2.575), (5, 2.9), (2.6, 2.9), (2.6, 3.275)])
    arrow(ax, [(5, 2.9), (7.4, 2.9), (7.4, 3.275)])
    box(ax, 2.6, 3.7, 4.0, '保留原件：源编码器', 'LiTS权重冻结，训练时仍计算')
    box(ax, 7.4, 3.7, 4.0, '复制一份：微调编码器', '用HCC分割损失更新权重', 'copy')
    text(ax, 5, 4.45, 'HCC适配时共同学习：微调副本、通道比例、解码器', 14)
    ax.plot([.55, 9.45], [4.9, 4.9], color='#cbd4de', lw=1)
    text(ax, 5, 5.35, '② 新病例预测：使用已经保存的两份权重', 17, True)
    box(ax, 5, 6.1, 4.0, '同一张新CT', '两路使用相同裁剪和空间坐标', 'out')
    arrow(ax, [(5, 6.525), (5, 6.85), (2.6, 6.85), (2.6, 7.225)])
    arrow(ax, [(5, 6.85), (7.4, 6.85), (7.4, 7.225)])
    box(ax, 2.6, 7.65, 4.0, '源编码器：原件', '保存的LiTS权重 → 四层原特征')
    box(ax, 7.4, 7.65, 4.0, '微调编码器：副本', '保存的HCC微调权重 → 四层新特征', 'copy')
    arrow(ax, [(2.6, 8.075), (2.6, 8.5), (3.2, 8.5), (3.2, 8.925)])
    arrow(ax, [(7.4, 8.075), (7.4, 8.5), (6.8, 8.5), (6.8, 8.925)])
    box(ax, 5, 9.35, 6.7, '四条skip分别逐通道混合', '每层各有一组比例；训练后固定，内部见图10', 'mix')
    arrow(ax, [(5, 9.775), (5, 10.325)])
    box(ax, 5, 10.75, 6.7, '解码与输出头 → 分割', '逐级：上采样 → 与同层混合skip相加 → 解码块', 'out')
    text(ax, 5, 11.55, '预测时两路都运行、全部权重固定；瓶颈来源及其更新范围仍待定', 13)
    save(fig, ax, '图9_方案5完整网络与通道混合位置')

def mechanism():
    fig, ax = start('图10  一条skip怎样混合原件与副本的特征', 9.5)
    text(ax, 5, .9, '对应图9的混合模块；同结构、同CT、同层、同位置', 14)
    box(ax, 2.6, 1.9, 4.1, '冻结原件输出 F_src', '[B,C,D,H,W]')
    box(ax, 7.4, 1.9, 4.1, '微调副本输出 F_adapt', '[B,C,D,H,W]', 'copy')
    arrow(ax, [(2.6, 2.325), (2.6, 2.85), (3.1, 2.85), (3.1, 3.475)])
    arrow(ax, [(7.4, 2.325), (7.4, 2.85), (6.9, 2.85), (6.9, 3.475)])
    text(ax, 5, 2.9, 'a：每通道一个比例，范围0至1', 13)
    box(ax, 5, 3.9, 7.3, '混合特征 = (1−a) × F_src + a × F_adapt', 'a形状 [1,C,1,1,1]；训练后固定，空间位置共用', 'mix')
    arrow(ax, [(5, 4.325), (5, 5.175)])
    box(ax, 5, 5.6, 7.3, '与同级上采样特征 U 相加', 'U来自当前级UpBlock；相加前空间尺寸必须一致', 'out')
    arrow(ax, [(5, 6.025), (5, 6.675)])
    box(ax, 5, 7.1, 5.4, '进入当前级MedNeXt解码块', '再继续后续解码，输出头产生最终分割', 'out')
    text(ax, 5, 8.12, 'B：批大小；C：当前层通道数；D、H、W：空间网格尺寸', 13)
    text(ax, 5, 8.65, 'HCC分割损失更新副本、a和解码器；原件权重冻结', 14)
    text(ax, 5, 9.1, '候选设计，尚未实现或验证', 13)
    save(fig, ax, '图10_方案5逐通道特征混合内部')

if __name__ == '__main__':
    overview()
    mechanism()

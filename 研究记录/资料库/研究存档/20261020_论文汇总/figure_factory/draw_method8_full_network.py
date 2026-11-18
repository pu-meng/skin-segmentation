"""Detailed whole-network drawing of Method 8, an unimplemented candidate.

Backbone order follows MedNeXtMultiStageSkipMoE.forward(); one shared multi-output
module and historical feedback follow the Method 8 design, not runnable code.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

ROOT = Path(__file__).resolve().parents[1]
REG = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
BOLD = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
plt.rcParams['svg.fonttype'] = 'path'
INK, BLUE, PURPLE = '#344254', '#2f6fa3', '#7e57a6'
COLORS = {'feature': ('#cfe3f6', BLUE), 'block': ('#f6e8a6', '#a66a00'),
          'route': ('#e6d5f7', PURPLE), 'output': ('#cdefd8', '#2e7d5b')}
fig, ax = plt.subplots(figsize=(10, 13.34))
fig.subplots_adjust(0, 0, 1, 1)
ax.set(xlim=(0, 27), ylim=(0, 36))
ax.axis('off')
texts, boxes = [], []


def text(x, y, s, size=12, bold=False, color=INK):
    t = ax.text(x, y, s, ha='center', va='center', fontsize=size,
                fontproperties=BOLD if bold else REG, color=color,
                linespacing=1.4, zorder=5)
    texts.append(t)
    return t


def box(x, y, w, name, sub, kind='block', h=1.5):
    fill, edge = COLORS[kind]
    p = FancyBboxPatch((x-w/2, y-h/2), w, h,
                       boxstyle='round,pad=0,rounding_size=0.12',
                       facecolor=fill, edgecolor=edge, lw=1.2, zorder=3)
    ax.add_patch(p)
    ts = [text(x, y+.27, name, 13, True), text(x, y-.32, sub, 11.8)]
    boxes.append((p, ts))


def arrow(a, b, color=INK, dashed=False):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=12,
                 lw=1.3, color=color, linestyle='--' if dashed else '-',
                 shrinkA=0, shrinkB=0, zorder=2))


def path(points, color=INK, dashed=False):
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=1.3,
                ls='--' if dashed else '-', zorder=1)
    arrow(points[-2], points[-1], color, dashed)


def plus(x, y):
    ax.add_patch(Circle((x, y), .25, facecolor='white', edgecolor=INK, lw=1.2, zorder=3))
    text(x, y, '+', 15)


text(13.5, 35.3, '方法8｜全部编码层 → 一个统一MoE → 四层解码', 17, True)
text(13.5, 34.5, '候选设计，尚未实现；中央仅一个模块，共享专家F与路由器G', 12)
box(3.6, 32.4, 5.2, 'CT → 输入投影', '[B,1,128,128,128]', 'feature')
box(12.7, 32.4, 7.0, '上一轮路径历史收益 H', '16条来源→接收路径分别计分', 'route')
box(23, 32.4, 5.2, '输出头 → 三类分割', '[B,3,128,128,128]', 'output')
arrow((3.6,31.65),(3.6,28.75))
arrow((23,28.75),(23,31.65))
ys = [28, 22.5, 17, 11.5]
enc_counts, dec_counts = [3,4,8,8], [3,4,8,8]
choices = {0:'E0、E1', 1:'E0、E1、E2', 2:'E1、E2、E3', 3:'E2、E3'}
# One physical shared module spans all source inputs and decoder outlets.
fill, edge = COLORS['route']
ax.add_patch(FancyBboxPatch((8.7,10.2),8.0,20.0,
    boxstyle='round,pad=0,rounding_size=0.16', facecolor=fill,
    edgecolor=edge,lw=1.6,zorder=3))
text(12.7,29.45,'统一MoE：四层共同使用',14,True,color=PURPLE)
text(12.7,26.4,'E0—E3分别投影至32通道',12)
text(12.7,25.25,'同一个专家 F：32→8→32',12,True)
text(12.7,24.1,'缓存四路修正 R0—R3',12)
text(12.7,21.05,'同一个路由器 G：全出口复用',12,True)
text(12.7,19.8,'读取来源摘要与当前 Uj',12)
text(12.7,18.55,'当前分数＋各路径历史偏好',12)
text(12.7,15.6,'按出口逐通道混合四路来源',12)
text(12.7,14.35,'对齐网格 → 输出投影 Qj',12)
text(12.7,12.9,'输出修正 ΔEj，恢复接收通道',12)
text(12.7,10.85,'4个出口；没有4套独立专家',12,True,color=PURPLE)
arrow((12.7,31.65),(12.7,30.2),PURPLE,True)
for j, y in enumerate(ys):
    c, n = 32*2**j, 128//2**j
    box(3.6,y,5.2,f'编码层 {j} ×{enc_counts[j]}',f'E{j}：[B,{c},{n},{n},{n}]')
    box(3.6,y-2.6,5.2,f'下采样 {j}：stride 2',f'{c} → {c*2}通道')
    arrow((3.6,y-.75),(3.6,y-1.85))
    if j<3:
        arrow((3.6,y-3.35),(3.6,ys[j+1]+.75))
    arrow((6.2,y),(8.7,y),BLUE)
    text(7.45,y+.42,f'E{j}',11.8,color=BLUE)
    plus(18.6,y)
    arrow((16.7,y),(18.35,y))
    text(17.5,y+.45,f'ΔE{j}',11.8)
    text(18.6,y+1.25,f'原skip E{j}',11.5,color=BLUE)
    arrow((18.6,y+.95),(18.6,y+.25),BLUE)
    box(23,y,5.2,f'解码层 {j} ×{dec_counts[j]}',f'[B,{c},{n},{n},{n}]')
    arrow((18.85,y),(20.4,y))
    box(23,y-2.6,5.2,f'上采样 {j}：stride 2',f'U{j}：[B,{c},{n},{n},{n}]')
    path([(23,y-1.85),(23,y-1.35),(18.6,y-1.35),(18.6,y-.25)])
    # Uj branches before addition and supplies the current-feature router.
    path([(19.5,y-1.35),(17.6,y-1.35),(17.6,y-.8),(16.7,y-.8)],BLUE)
    text(18.6,y-1.85,f'U{j} → 同一G',10.5,color=BLUE)
    if j<3:
        arrow((23,ys[j+1]+.75),(23,y-3.35))
box(3.6,6.3,5.2,'卷积瓶颈 ×8','[B,512,8,8,8]')
arrow((3.6,8.15),(3.6,7.05))
box(3.6,3.9,5.2,'原瓶颈上下文','MHA＋MoE，保留主干')
arrow((3.6,5.55),(3.6,4.65))
path([(3.6,3.15),(3.6,2.4),(23,2.4),(23,8.15)])
text(13.4,6.7,'共享参数：1个F＋1个G',13,True,color=PURPLE)
text(13.4,5.55,'4个输入投影；4个接收投影；4个输出投影',11.8)
text(13.4,4.4,'每层：Uj＋原skip Ej＋模块输出 ΔEj → Dj',12)
text(13.5,1.55,'蓝线：特征；紫虚线：历史状态；原skip标签表示缓存的同层编码特征。',12)
text(13.5,.7,'顺序：共享处理并缓存 → D3出口 → D2出口 → D1出口 → D0出口。',12)


def main():
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for t in texts:
        bb = t.get_window_extent(renderer)
        assert fig.bbox.contains(bb.x0,bb.y0) and fig.bbox.contains(bb.x1,bb.y1), t.get_text()
    for p, ts in boxes:
        bb = p.get_window_extent(renderer)
        for t in ts:
            tb = t.get_window_extent(renderer)
            assert tb.x0>bb.x0 and tb.x1<bb.x1 and tb.y0>bb.y0 and tb.y1<bb.y1, t.get_text()
    output = ROOT/'figures'/'方法8_统一MoE_完整网络总图.svg'
    fig.savefig(output, facecolor='white', metadata={'Date':None})
    ET.parse(output)
    preview = Path('/tmp/method8_full_network_preview.png')
    fig.savefig(preview, dpi=150, facecolor='white')
    plt.close(fig)
    print(output)
    print(preview)


if __name__ == '__main__':
    main()

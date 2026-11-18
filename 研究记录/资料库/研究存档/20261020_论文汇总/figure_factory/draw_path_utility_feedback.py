"""Draw an unimplemented candidate; does not change networks or launch training.

SVGs are the deliverables. PNGs in /tmp are only visual-review previews.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures'
PREVIEW = Path('/tmp/path_utility_feedback_preview')
REG = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
BOLD = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
plt.rcParams['svg.fonttype'] = 'path'
INK = '#344254'
BLUE = '#2f6fa3'
PURPLE = '#7e57a6'
COLORS = dict(base=('#f6e8a6', '#a66a00'), feature=('#cfe3f6', BLUE),
              route=('#e6d5f7', PURPLE), out=('#cdefd8', '#2e7d5b'),
              plain=('#e7e9ed', '#667085'))


def start(height, title, subtitle):
    global fig, ax, text_checks, box_checks
    fig, ax = plt.subplots(figsize=(7.2, height * .4))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0, 18), ylim=(0, height))
    ax.axis('off')
    text_checks, box_checks = [], []
    txt(9, height-.65, title, 13, bold=True)
    txt(9, height-1.4, subtitle, 9.5)


def txt(x, y, value, size=10.5, color=INK, bold=False):
    t = ax.text(x, y, value, ha='center', va='center', fontsize=size,
                fontproperties=BOLD if bold else REG, color=color,
                linespacing=1.4, zorder=5)
    text_checks.append(t)
    return t


def box(x, y, width, title, sub='', kind='base', height=1.3):
    fill, edge = COLORS[kind]
    p = FancyBboxPatch((x-width/2, y-height/2), width, height,
        boxstyle='round,pad=0,rounding_size=0.12', lw=1.2,
        facecolor=fill, edgecolor=edge, zorder=3)
    ax.add_patch(p)
    ts = [txt(x, y+.23 if sub else y, title, 10.5, bold=True)]
    if sub:
        ts.append(txt(x, y-.28, sub, 9.2))
    box_checks.append((p, ts))


def arrow(a, b, color=INK, dashed=False):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=10,
        lw=1.2, color=color, linestyle='--' if dashed else '-',
        shrinkA=0, shrinkB=0, zorder=2))


def path(points, color=INK, dashed=False):
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color,
                lw=1.2, ls='--' if dashed else '-', zorder=1)
    arrow(points[-2], points[-1], color, dashed)


def plus(x, y):
    ax.add_patch(Circle((x, y), .23, facecolor='white', edgecolor=INK, lw=1.2, zorder=3))
    txt(x, y, '+', 13)


def save(name):
    OUT.mkdir(exist_ok=True)
    PREVIEW.mkdir(exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for t in text_checks:
        bb = t.get_window_extent(renderer)
        if not fig.bbox.contains(bb.x0, bb.y0) or not fig.bbox.contains(bb.x1, bb.y1):
            raise RuntimeError('Text outside canvas: '+t.get_text())
    for p, ts in box_checks:
        bb = p.get_window_extent(renderer)
        for t in ts:
            tb = t.get_window_extent(renderer)
            if tb.x0 < bb.x0+2 or tb.x1 > bb.x1-2 or tb.y0 < bb.y0 or tb.y1 > bb.y1:
                raise RuntimeError('Text outside box: '+t.get_text())
    svg = OUT / (name+'.svg')
    fig.savefig(svg, facecolor='white', metadata={'Date': None})
    ET.parse(svg)
    fig.savefig(PREVIEW / (name+'.png'), dpi=150, facecolor='white')
    plt.close(fig)
    print(svg)


def overview():
    start(25, '图1｜全部编码输入一个统一MoE，输出到四层解码',
          '候选；一个共享专家F、一个共享路由器G；16条来源→接收路径')
    for i,x in enumerate((2.3,6.7,11.1,15.5)):
        box(x,21.7,3.6,f'编码 E{i}',f'独立投影 P{i}','feature')
        arrow((x,21.05),(x,19.5),BLUE)
    fill, edge = COLORS['route']
    ax.add_patch(FancyBboxPatch((.9,13.5),16.2,6,boxstyle='round,pad=0,rounding_size=0.12',facecolor=fill,edgecolor=edge,lw=1.3,zorder=3))
    txt(9,18.5,'一个统一MoE模块',12,bold=True)
    txt(9,17.3,'同一个F处理四路输入 → 缓存R0—R3；全部出口复用',10)
    txt(9,16.1,'同一个G读取当前Uj与来源摘要 → 各通道四路权重',10)
    txt(9,14.9,'每个出口：对齐四路R → 加权混合 → 输出投影Qj',10)
    for j,x in zip((3,2,1,0),(2.3,6.7,11.1,15.5)):
        arrow((x,13.5),(x,11.1),PURPLE)
        txt(x,12.3,f'出口j={j}：ΔE{j}',9)
        box(x,10.3,3.6,f'加回原 E{j}',f'再与 U{j} 相加','feature',1.6)
        arrow((x,9.5),(x,7.85))
        box(x,7.2,3.6,f'解码 D{j}',f'处理 U{j}＋E{j}′','base')
    txt(9,5.4,'输出按D3→D2→D1→D0依次产生；Uj由当前UpBlock提供给同一个G',9.4)
    txt(9,4.2,'每个出口都读取E0—E3；出口有独立投影，没有独立专家',10)
    txt(9,2.9,'H：每个来源到每个出口分别计分，共16个历史状态',10,color=PURPLE)
    txt(9,1.5,'本图展开共享关系；完整主干顺序与Uj反馈连线见总图',9.5)
    save('方法8_统一MoE_四层总览')


def combine():
    start(29,'图2｜统一模块内部：共享处理，按出口路由',
          '以D2出口为例；四个出口共用同一个专家F和同一个路由器G')
    box(9,25.8,15,'E0—E3 → 各自输入投影Pi → Z0—Z3',
        '统一32通道；各来源保留原空间网格','feature',1.5)
    box(9,22.7,15,'一个共享专家F：分别处理Z0—Z3',
        '得到并缓存R0—R3；所有出口复用，不设置F0—F3','route',1.5)
    arrow((9,25.05),(9,23.45),BLUE)
    box(5,19.6,7.6,'一个共享路由器G',
        '来源摘要＋投影后的U2','route',1.5)
    box(13.6,19.6,6.2,'H(i,2) → 偏好b',
        '四路分别计分','route',1.5)
    path([(9,25.05),(.65,25.05),(.65,19.6),(1.2,19.6)],BLUE)
    txt(4.8,20.9,'U2池化 → 接收投影A2 → G',9.1,BLUE)
    box(9,16.6,15,'当前分数a＋λb → 沿四路来源做softmax',
        '每个共享通道四个比例；权重形状[B,4,32]','route',1.5)
    path([(5,18.85),(5,18.1),(6,18.1),(6,17.35)],PURPLE)
    path([(13.6,18.85),(13.6,18.1),(12,18.1),(12,17.35)],PURPLE)
    box(9,13.4,15,'R0—R3对齐到E2网格，逐通道加权相加',
        'V2＝w02×T02＋w12×T12＋w22×T22＋w32×T32','route',1.5)
    arrow((9,15.85),(9,14.15),PURPLE)
    path([(16.5,22.7),(17.35,22.7),(17.35,13.4),(16.5,13.4)],BLUE)
    box(9,10.5,12,'输出投影Q2：32→128通道',
        '得到与原skip E2形状一致的修正ΔE2','feature',1.5)
    arrow((9,12.65),(9,11.25))
    plus(9,7.8)
    arrow((9,9.75),(9,8.03))
    box(3.4,7.8,4.8,'原skip E2','保留直连','feature')
    arrow((5.8,7.8),(8.77,7.8),BLUE)
    plus(9,5.7)
    arrow((9,7.57),(9,5.93))
    box(14.4,5.7,4.8,'接收特征U2','保留主路径','feature')
    arrow((12,5.7),(9.23,5.7),BLUE)
    box(9,3.5,12,'Decoder Stage 2 → 后续解码 → 分割',
        '输入为U2＋E2＋ΔE2','out',1.5)
    arrow((9,5.47),(9,4.25))
    txt(9,1.9,'同一个Ri可送到其他出口；关闭(i,2)只关闭D2这一条加权路径',9.2)
    txt(9,.85,'逐通道路由在32通道共享空间执行；Qj恢复不同出口的实际通道数',9.2)
    save('方法8_统一MoE_双分数融合')


def feedback():
    start(29, '图3｜第 n 轮测路径收益，第 n＋1 轮反馈给路由',
          '只用训练样本评分；同批CT、同标注、同参数；此流程尚未实现')
    box(9,25.8,13.8,'抽取一次训练批次，选择一条路径 E_i → D_j',
        '本轮历史分数保持固定；评分前不更新模型参数','feature',1.5)
    box(4.6,22.8,7.2,'参照：全部路径开启','正常解码 → 分割 P_all','out',1.5)
    box(13.3,22.8,6.5,'比较：仅关闭指定路径','该路加权特征置零 → P_off','out',1.5)
    path([(9,25.05),(9,24.4),(4.6,24.4),(4.6,23.55)])
    path([(9,24.4),(13.3,24.4),(13.3,23.55)])
    box(4.6,19.8,7.2,'与同一份真实标注比较','得到损失 L_all','plain')
    box(13.3,19.8,6.5,'与同一份真实标注比较','得到损失 L_off','plain')
    arrow((4.6,22.05),(4.6,20.45))
    arrow((13.3,22.05),(13.3,20.45))
    box(9,16.8,13.8,'该路径本次收益 g＝L_off − L_all',
        '正值：关闭后更差，加分；负值：关闭后更好，减分','route',1.5)
    path([(4.6,19.15),(4.6,18.2),(6,18.2),(6,17.55)])
    path([(13.3,19.15),(13.3,18.2),(12,18.2),(12,17.55)])
    box(9,13.8,13.8,'在第 n 轮内轮换路径，积累各自收益',
        '本页示例：250训练批；每10批评分1条路径，共25次','route',1.5)
    arrow((9,16.05),(9,14.55),PURPLE)
    box(9,10.8,13.8,'轮末：逐路径取均值并平滑 → 更新 H',
        '各路径分别计数；未被评分的路径保留旧值','route',1.5)
    arrow((9,13.05),(9,11.55),PURPLE)
    box(9,7.8,13.8,'缩放并限制历史偏好的幅度 → b',
        '保留正负号；停止梯度；避免压过当前特征分数','route',1.5)
    arrow((9,10.05),(9,8.55),PURPLE)
    box(9,4.8,13.8,'第 n＋1 轮：当前分数 a＋λ×b → 路由权重',
        '新一轮仍正常训练；下一轮末再更新历史分数','route',1.5)
    arrow((9,7.05),(9,5.55),PURPLE)
    txt(9,2.8,'评分比较只前向，不反传；正常参照分支仍按原流程反向更新参数',9.3)
    txt(9,1.6,'推理：冻结训练结束时的 H；仅输入CT，不用真实标注，不在线评分',9.3)
    save('方法8_统一MoE_轮次更新')


if __name__ == '__main__':
    overview()
    combine()
    feedback()

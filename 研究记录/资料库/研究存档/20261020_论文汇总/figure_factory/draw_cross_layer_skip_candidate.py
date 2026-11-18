"""Discussion figures only; does not implement or modify the training network."""
from pathlib import Path
import ast
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[1]
FONT = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['svg.fonttype'] = 'path'
INK = '#28384A'
BLUE = '#3478A5'
COL = {'base': '#F6E8A6', 'feature': '#E8F2FC', 'new': '#E6D5F7',
       'weight': '#FFF0D8', 'out': '#CDEFD8', 'context': '#CFE3F6'}

def start(w, h, inches):
    global fig, ax
    fig, ax = plt.subplots(figsize=inches)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0, w), ylim=(0, h))
    ax.axis('off')

def txt(x, y, s, size=16, color=INK):
    ax.text(x, y, s, fontproperties=FONT, fontsize=size, ha='center',
            va='center', color=color, linespacing=1.3, zorder=5)

def box(x, y, w, title, sub='', kind='base', h=.92):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
        boxstyle='round,pad=0.02,rounding_size=0.09',
        facecolor=COL[kind], edgecolor=INK, lw=1.3, zorder=3))
    txt(x, y+.17 if sub else y, title, 17)
    if sub:
        txt(x, y-.23, sub, 14)

def arr(a, b, color=INK, dashed=False):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=13,
        lw=1.5, color=color, linestyle='--' if dashed else '-',
        shrinkA=0, shrinkB=0, zorder=2))

def path(points, color=INK, dashed=False):
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=1.5,
                ls='--' if dashed else '-', zorder=1)
    arr(points[-2], points[-1], color, dashed)

def plus(x, y):
    ax.add_patch(Circle((x,y), .19, facecolor='white', edgecolor=INK, lw=1.3, zorder=3))
    txt(x,y,'+',18)

def save(name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for t in ax.texts:
        b = t.get_window_extent(renderer)
        if b.x0 < 0 or b.y0 < 0 or b.x1 > fig.bbox.x1 or b.y1 > fig.bbox.y1:
            raise RuntimeError('Text outside canvas: '+t.get_text())
    out = ROOT / 'figures' / (name+'.svg')
    fig.savefig(out, facecolor='white', metadata={'Date': None})
    ET.parse(out)
    fig.savefig('/tmp/'+name+'.png', dpi=90, facecolor='white')
    plt.close(fig)
    print(out)

def overview():
    start(19, 26, (15, 20))
    txt(9.5,25.4,'01-C(a)｜四层 skip 跨层加权：完整网络候选',23)
    txt(9.5,24.75,'沿用 01(a) 的 MedNeXt 骨架；紫色为拟新增模块，尚未实现或训练',15)
    box(3,23.5,3.2,'CT patch','[B,1,128,128,128]','feature')
    box(3,21.9,3.2,'Stem','1 → 32')
    arr((3,23.04),(3,22.36))
    ys = [19.6,16.6,13.6,10.6]
    counts = [3,4,8,8]
    box(9.7,22.8,5.1,'四层特征集合 E0、E1、E2、E3','只收集 / 缓存，不在此处相加','feature',1.2)
    txt(14.9,22.8,'同一编码器的四层输出\n本候选不使用 Swin',16)
    arr((3,21.44),(3,20.06))
    for i, y in enumerate(ys):
        c, s = 32*2**i, 128//2**i
        box(3,y,3.2,f'Encoder Stage {i}',f'{counts[i]} blocks | {c} ch, {s}³')
        # Separate orthogonal collection lanes, all cached before decoding starts.
        lane = 5.1+.46*i
        porty = 22.42+.24*i
        path([(4.6,y),(lane,y),(lane,porty),(7.15,porty)], BLUE)
        txt(4.98,y+.3,f'E{i}',14,BLUE)
        if i < 3:
            box(3,y-1.5,2.65,f'DownBlock {i}','stride 2',h=.85)
            arr((3,y-.46),(3,y-1.075))
            arr((3,y-1.925),(3,ys[i+1]+.46))
        box(10,y,4.0,f'跨层融合 F{i}',f'读取 E0、E1、E2、E3','new',1.05)
        # Bundle distribution is explicitly a tuple, not an aggregate tensor.
        path([(7.25,y),(8,y)],BLUE)
        plus(13.1,y)
        arr((12,y),(12.91,y))
        arr((13.29,y),(14.5,y))
        box(16.1,y,3.2,f'Decoder Stage {i}',f'{counts[i]} blocks | {c} ch, {s}³')
        if i < 3:
            upy = y-1.5
            box(16.1,upy,2.7,f'UpBlock {i}','stride 2',h=.85)
            arr((16.1,ys[i+1]+.46),(16.1,upy-.425))
            path([(16.1,upy+.425),(16.1,y-.85),(13.1,y-.85),(13.1,y-.19)])
    path([(9.7,22.2),(9.7,21.5),(7.25,21.5),(7.25,10.6)],BLUE)
    txt(10,20.8,'每层融合内部见图(b)',14)
    box(3,9.1,2.65,'DownBlock 3','256 → 512',h=.85)
    arr((3,10.14),(3,9.525))
    box(3,7.5,3.4,'Conv bottleneck','8 blocks | 512 ch, 8³')
    arr((3,8.675),(3,7.96))
    box(3,5.8,4.4,'原模型上下文模块','保留原 MHA＋MoE','context',1.0)
    arr((3,7.04),(3,6.3))
    box(16.1,7.5,3.2,'UpBlock 3','512 → 256, 16³')
    path([(3,5.3),(3,4.3),(16.1,4.3),(16.1,7.04)])
    path([(16.1,7.96),(16.1,8.9),(13.1,8.9),(13.1,10.41)])
    txt(15,9.3,'U3',14)
    box(16.1,21.4,3.2,'分割输出头','背景 / 肝脏 / 肿瘤','out')
    arr((16.1,20.06),(16.1,20.94))
    txt(9.5,3.25,'初始化：同层 aii＝1；跨层 aij＝0（i ≠ j）',19)
    txt(9.5,2.45,'Fi＝aii × Ei ＋ 三条跨层加权特征；Fi 与 Ui 相加后进入 Decoder Stage i',15)
    txt(9.5,1.65,'蓝色集合线表示四个独立张量的分发；不是拼接、求和或新的特征网络',15)
    txt(9.5,.85,'主干、瓶颈及解码器接法据 mednext.py 核对；仅展示最终输出头，省略辅助监督头',14)
    save('01C_a_跨层skip完整网络候选')

def detail():
    start(18,15,(15,12.5))
    txt(9,14.45,'01-C(b)｜展开 F3：每条跨层路径只用一个卷积',23)
    txt(9,13.75,'示例：目标为第3层，256通道、16³；四路独立并联，不串联处理',16)
    xs=[2.4,6.8,11.2,15.6]
    for j,x in enumerate(xs):
        box(x,12.5,3.8,f'编码器特征 E{j}',f'[B,{32*2**j},{128//2**j}³]','feature')
        if j != 3:
            box(x,10.8,3.8,'空间对齐','平均池化 → 16³','new')
            arr((x,12.04),(x,11.26))
            box(x,9.1,3.8,'单个 1×1×1 Conv',f'{32*2**j} → 256 ch','new')
            arr((x,10.34),(x,9.56))
            box(x,7.4,3.8,f'乘标量 a3{j}','初始值 0','weight')
            arr((x,8.64),(x,7.86))
        else:
            box(x,9.1,3.8,'原同层直连','Identity：不加卷积','feature')
            arr((x,12.04),(x,9.56))
            box(x,7.4,3.8,'乘标量 a33','初始值 1','weight')
            arr((x,8.64),(x,7.86))
    box(9,5.4,13.5,'四路逐元素相加 → F3','[B,256,16,16,16]；不是通道拼接','new',1.05)
    for x in xs:
        arr((x,6.94),(x,5.925))
    plus(9,3.8)
    arr((9,4.875),(9,3.99))
    box(14.5,3.8,4.3,'UpBlock 3 输出 U3','[B,256,16,16,16]','feature')
    arr((12.35,3.8),(9.19,3.8))
    box(9,2.3,7,'Decoder Stage 3','输入 F3＋U3；保留原解码块')
    arr((9,3.61),(9,2.76))
    txt(9,1.15,'其他 Fi 同理：向浅层传递时用三线性上采样；向深层传递时用平均池化',15)
    txt(9,.5,'16个标量＋12个跨层投影卷积；全局标量不随病例或位置变化；无 top-k 路由',14)
    save('01C_b_单层四路融合与初始化')

if __name__ == '__main__':
    ast.parse(Path(__file__).read_text())
    overview()
    detail()

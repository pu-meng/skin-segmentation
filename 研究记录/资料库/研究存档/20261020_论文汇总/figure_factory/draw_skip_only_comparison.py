"""Reuse Method6 overview and drawing primitives; current skip-only code checked."""
from pathlib import Path
import json
import sys
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import draw_cross_layer_skip_candidate as g
import draw_multistage_moe_candidate as old

OUT = g.ROOT / 'figures'
NAMES = []

# Typography is checked at normal Markdown width, not zoomed SVG inspection.
ORIGINAL_TXT = g.txt
BOX_TEXT_AUDIT = []

def readable_txt(x, y, text, size=16, color=g.INK):
    return ORIGINAL_TXT(x, y, text, max(20, size * 1.20), color)


def readable_box(x, y, w, title, sub='', kind='base', h=.92):
    g.ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,
        boxstyle='round,pad=0.02,rounding_size=0.09', facecolor=g.COL[kind],
        edgecolor=g.INK,lw=1.3,zorder=3))
    # Measure rendered glyphs and use the available width AND height.
    fig = g.fig
    renderer = fig.canvas.get_renderer()
    x0,y0 = g.ax.transData.transform((x-w/2,y-h/2))
    x1,y1 = g.ax.transData.transform((x+w/2,y+h/2))
    tw, th = x1-x0, y1-y0
    fonts = [28 if w<8 else 32, 23 if w<8 else 26]
    artists = []
    for i, value in enumerate((title,sub) if sub else (title,)):
        artist=g.ax.text(x,y,value,fontproperties=g.FONT,fontsize=fonts[i],
            ha='center',va='center',color=g.INK,zorder=5,linespacing=1.1)
        box=artist.get_window_extent(renderer)
        if box.width>tw*.90:
            artist.set_fontsize(artist.get_fontsize()*tw*.90/box.width)
        artists.append(artist)
    gap=fig.dpi*2/72
    boxes=[a.get_window_extent(renderer) for a in artists]
    total=sum(b.height for b in boxes)+(gap if sub else 0)
    if total>th*.88:
        for artist in artists:
            artist.set_fontsize(artist.get_fontsize()*(th*.88-(gap if sub else 0))/sum(b.height for b in boxes))
    boxes=[a.get_window_extent(renderer) for a in artists]
    px_per_y=g.ax.transData.transform((0,1))[1]-g.ax.transData.transform((0,0))[1]
    if sub:
        artists[0].set_y(y+(boxes[1].height+gap)/2/px_per_y)
        artists[1].set_y(y-(boxes[0].height+gap)/2/px_per_y)
    for artist in artists:
        bounds=artist.get_window_extent(renderer)
        if bounds.x0<x0 or bounds.x1>x1 or bounds.y0<y0 or bounds.y1>y1:
            raise RuntimeError('Text spills outside its box: '+artist.get_text())
        BOX_TEXT_AUDIT.append(dict(text=artist.get_text(), source_font_pt=round(artist.get_fontsize(),2),
            screen_font_px_at_800=round(artist.get_fontsize()*800/(fig.get_size_inches()[0]*72),2)))


g.txt, g.box = readable_txt, readable_box

def save(name):
    g.fig.canvas.draw()
    renderer = g.fig.canvas.get_renderer()
    for text in g.ax.texts:
        bounds = text.get_window_extent(renderer)
        if bounds.x0 < 0 or bounds.y0 < 0 or bounds.x1 > g.fig.bbox.x1 or bounds.y1 > g.fig.bbox.y1:
            raise RuntimeError('Text outside canvas: ' + text.get_text())
    path = OUT / (name + '.svg')
    g.fig.savefig(path, facecolor='white', metadata={'Date': None})
    ET.parse(path)
    # Actual reading-width preview, not an extra permanent deliverable.
    g.fig.savefig('/tmp/' + name + '.png', dpi=800 / g.fig.get_size_inches()[0], facecolor='white')
    plt.close(g.fig)
    NAMES.append(name)


def overview(mode):
    # Reuse the original geometry, but render one explicit model per figure.
    original = {name: getattr(g, name) for name in ('txt', 'save', 'box', 'arr', 'path')}
    label = {'A05': '无新增融合', 'C13': '全局固定权重融合', 'C14': '逐位置动态权重融合'}[mode]
    mapping = {
        '图7｜方法6：多stage MoE修正skip 2': f'图1｜{mode}：{label}',
        '基础结构已实现；训练与评估状态见实验记录；紫色为新增模块': '本图只画A05；skip 2直接传原E2，没有紫色模块' if mode == 'A05' else f'本图只画{mode}；紫色模块是新增skip补充，不是另一套网络',
        '原瓶颈上下文': '瓶颈上下文',
        'CT → Input Projection': 'CT→输入投影',
        '示例输入128³，基础通道32': '128³输入／32通道',
        'Conv bottleneck': '卷积瓶颈',
        '三张独立特征图，不在此处融合': '收集三路特征，暂不混合',
        '保留已有 MHA＋MoE': 'MHA＋普通FFN',
        '新增MoE修正模块': f'{mode}：' + ('固定权重融合' if mode == 'C13' else '动态权重融合'),
        'E1、E2、E3、U2 → E2′': '三路补充×全局比例' if mode == 'C13' else '三路补充×逐位置比例',
        'U2：解码上下文': 'U2：用于计算动态权重',
        '只修改skip 2：E2′＝E2＋加权专家修正；Decoder输入仍为U2＋E2′': 'A05：原E2直接与U2相加，再进入Decoder 2' if mode == 'A05' else f'{mode}：E2′＝原E2＋三路加权补充；Decoder 2接收U2＋E2′',
        'skip 0、1、3保留；编码器与解码器均运行一次；局部模块见图8': 'A05没有新增专家、评分器或补充特征' if mode == 'A05' else ('C13：权重随训练学习；预测时所有患者、位置共用一组' if mode == 'C13' else 'C14：评分网络参数训练后固定；权重随患者和位置变化'),
        '现有方法6已缓存E3至skip 2使用之后；原skip 3与瓶颈上下文保留': '三组瓶颈均为MHA＋普通FFN；其他skip保持原样',
        '主干按实际forward核对；省略辅助监督头；此图仅说明基础版结构': '沿用旧图布局并按当前代码修订；三路专家与加权过程见后续图',
    }
    def txt(x, y, text, size=16, color=g.INK):
        if mode != 'C14' and text == 'U2：解码上下文':
            return
        text=mapping.get(text,text)
        text=text.replace('Encoder Stage ', '编码层 ').replace('Decoder Stage ', '解码层 ')
        text=text.replace('DownBlock ', '下采样 ').replace('UpBlock ', '上采样 ')
        return original['txt'](x, y, text, size, color)
    def box(x, y, w, title, sub='', kind='base', h=.92):
        if mode == 'A05' and title in ('缓存 E1、E2、E3', '新增MoE修正模块'):
            return
        title=mapping.get(title,title)
        title=title.replace('Encoder Stage ', '编码层 ').replace('Decoder Stage ', '解码层 ').replace('DownBlock ', '下采样 ').replace('UpBlock ', '上采样 ')
        return original['box'](x, y, w, title, mapping.get(sub,sub), kind, h)
    def arr(a, b, color=g.INK, dashed=False):
        if mode == 'A05' and a == (11.3, 14):
            a = (4.6, 14)
        return original['arr'](a, b, color, dashed)
    def path(points, color=g.INK, dashed=False):
        # Remove collection and correction input paths when no module exists.
        if mode == 'A05' and color == g.BLUE:
            return
        # Decoder context is a learned scoring input only for dynamic C14.
        if mode != 'C14' and points[0] == (13.6, 13.15):
            return
        return original['path'](points, color, dashed)
    def finish(unused):
        if mode == 'A05':
            original['txt'](9, 14.4, 'A05：原E2直接传递', 15)
        save(f'skip候选_01_{mode}_独立网络')
    g.txt, g.box, g.arr, g.path, g.save = txt, box, arr, path, finish
    try:
        old.overview()
    finally:
        for name, value in original.items():
            setattr(g, name, value)


def experts():
    g.start(20, 24, (12, 14.4))
    g.txt(10,23.3,'图2｜C13与C14共用：三路特征变成三份补充',21)
    g.txt(10,22.5,'同一张CT、同一个编码器；每路参数独立，三个专家全部计算',14)
    xs = [3.5,10,16.5]
    for j,x in enumerate(xs,1):
        source = {1:'第2级编码特征 E1',2:'第3级编码特征 E2',3:'第4级编码特征 E3'}[j]
        g.box(x,20.8,5.7,source,f'{32*2**j}通道，{128//2**j}³网格','feature',1.2)
        g.arr((x,20.2),(x,18.9))
        title = {1:'平均池化 → 32³',2:'保持32³网格',3:'三线性插值 → 32³'}[j]
        g.box(x,18.3,5.7,title,'先对齐空间尺寸','new',1.2)
        g.arr((x,17.7),(x,16.4))
        g.box(x,15.8,5.7,'1×1×1卷积投影',f'{32*2**j} → 128通道，得到Z{j}','new',1.2)
        g.arr((x,15.2),(x,13.9))
        g.box(x,13.3,5.7,f'专家{j}：降维卷积','128 → 32通道','new',1.2)
        g.arr((x,12.7),(x,11.4))
        g.box(x,10.8,5.7,'GELU','对特征做非线性变换','new',1.2)
        g.arr((x,10.2),(x,8.9))
        g.box(x,8.3,5.7,'升维卷积','32 → 128通道','new',1.2)
        g.arr((x,7.7),(x,6.4))
        g.box(x,5.8,5.7,f'补充特征 R{j}','128通道，32³网格','out',1.2)
    g.box(10,3.3,16.8,'下一步：给R1、R2、R3乘权重，再求和','C13与C14只在权重生成方式上不同，见图3','weight',1.4)
    for x in xs:
        g.arr((x,5.2),(x,4.0))
    g.txt(10,1.8,'E2是原skip特征；Z2是投影后的特征；R2是专家生成的补充',14)
    g.txt(10,.8,'示例输入为128³；专家末层零初始化，起点不改变原网络输出',14)
    save('skip候选_02_三路专家')


def differences():
    g.start(20, 29, (12, 17.4))
    g.txt(10,28.3,'图3｜三组的区别：是否补充，以及谁决定权重',21)
    g.txt(10,27.5,'符号沿用图1、图2；示例权重只为说明，并非实验读数',14)
    # A05: explicit addition, not concatenation.
    g.txt(10,26.4,'A05｜无新增融合',19)
    g.box(4,24.8,5.6,'原skip特征 E2','直接传给解码器','feature',1.2)
    g.plus(10,24.8)
    g.arr((6.8,24.8),(9.81,24.8))
    g.box(16,24.8,5.6,'Decoder 2','继续后续解码','base',1.2)
    g.arr((10.19,24.8),(13.2,24.8))
    g.box(10,22.8,6.0,'上采样特征 U2','来自UpBlock 2','feature',1.0)
    g.arr((10,23.3),(10,24.61))
    # C13.
    g.txt(10,20.9,'C13｜固定融合：权重可学习，使用时不看输入',19)
    g.box(4.3,19.1,6.8,'三个全局可训练分数','初始均为0；随训练更新','weight',1.2)
    g.box(15.5,19.1,7.0,'对三个分数做softmax','得到全局权重w1、w2、w3','weight',1.2)
    g.arr((7.7,19.1),(12,19.1))
    g.box(10,16.7,17.6,'所有患者、所有空间位置，共用同一组权重','示例：患者甲和患者乙均使用 0.2、0.5、0.3','weight',1.3)
    g.path([(15.5,18.5),(15.5,17.35)])
    g.box(10,14.4,17.6,'补充量 ΔE2 = w1×R1 + w2×R2 + w3×R3','修正skip：E2′ = E2 + ΔE2；Decoder 2接收U2 + E2′','out',1.4)
    g.arr((10,16.05),(10,15.1))
    # C14.
    g.txt(10,12.4,'C14｜动态融合：当前特征决定当前位置的权重',19)
    g.box(10,10.6,17.6,'沿通道拼接 Z1、Z2、Z3、U2','三个对齐编码特征＋当前解码特征：128×4 = 512通道','feature',1.3)
    g.arr((10,9.95),(10,8.95))
    g.box(10,8.3,17.6,'评分网络：1³卷积512→32；GELU；1³卷积32→3','再沿三路softmax：每个空间位置得到三个权重','weight',1.3)
    g.arr((10,7.65),(10,6.65))
    g.box(10,6.0,17.6,'不同患者、不同空间位置，可以使用不同权重','示例：位置甲 0.7、0.2、0.1；位置乙 0.1、0.3、0.6','weight',1.3)
    g.arr((10,5.35),(10,4.35))
    g.box(10,3.7,17.6,'补充量 ΔE2 = w1×R1 + w2×R2 + w3×R3','修正skip：E2′ = E2 + ΔE2；Decoder 2接收U2 + E2′','out',1.3)
    g.txt(10,1.8,'两种融合都是三路全部计算、连续加权；没有共享专家或Top-k筛选',14)
    g.txt(10,.8,'所有相加、乘权重操作均逐元素进行；各位置三路权重之和为1',14)
    save('skip候选_03_固定与动态区别')


if __name__ == '__main__':
    for mode in ('A05', 'C13', 'C14'):
        overview(mode)
    experts()
    differences()
    Path(__file__).with_name('skip图_默认预览字号核验.json').write_text(json.dumps(BOX_TEXT_AUDIT, ensure_ascii=False, indent=2))
    print(json.dumps(NAMES, ensure_ascii=False))

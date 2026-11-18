"""Compact code-faithful architecture panels, readable at 800 px Markdown width.

Layout references: RMT/RMT.png; SU-RMT Fig.2; MambaLiteUNet Fig.2.
Local reference: 研究存档/20261020_论文汇总/figures/
图5_方案3完整冻结网络与外部记忆位置.png (encoder left, decoder right).
Only the layout is reused; all model operations follow Models/TinyViM.
"""
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Ellipse, Rectangle

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'log/TinyViM_频谱分析/整网结构'
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'svg.fonttype': 'none'})
INK='#293a4b'
BLUE='#e5eff9'
GREEN='#dff1e8'
YELLOW='#f7e9ab'
PURPLE='#e9dff3'
GRAY='#f1f3f5'

def canvas(h, title, width=1000):
    fig,ax=plt.subplots(figsize=(width/100,h/100), dpi=180)
    ax.set(xlim=(0,1000), ylim=(h,0)); ax.axis('off')
    fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
    text(ax,500,27,title,18,bold=True)
    return fig,ax

def text(ax,x,y,s,fs=14,ha='center',bold=False,color=INK):
    return ax.text(x,y,s,ha=ha,va='center',fontsize=fs*(ax.figure.get_figwidth()/10),color=color,
            weight='bold' if bold else 'normal',linespacing=1.35)

def box(ax,x,y,w,h,s,color=BLUE,fs=14):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=7',
                              facecolor=color,edgecolor=INK,linewidth=1.2))
    artist=text(ax,x+w/2,y+h/2,s,fs)
    if not hasattr(ax, "node_checks"): ax.node_checks=[]
    ax.node_checks.append((x,y,w,h,artist))

def path(ax,pts,color=INK,dashed=False):
    assert all(x == xx or y == yy for (x,y),(xx,yy) in zip(pts,pts[1:])), pts
    xs,ys=zip(*pts)
    ax.plot(xs,ys,color=color,lw=1.25,ls='--' if dashed else '-',zorder=1)

def arrow(ax,x,y,xx,yy,color=INK):
    assert x == xx or y == yy, (x,y,xx,yy)
    ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops=dict(arrowstyle='-|>',lw=1.25,color=color,shrinkA=0,shrinkB=2),zorder=2)

def route(ax,pts,color=INK):
    path(ax,pts[:-1],color)
    arrow(ax,*pts[-2],*pts[-1],color)

def op(ax,x,y,s,color=PURPLE,r=17):
    ax.add_patch(Ellipse((x,y),2*r/(ax.figure.get_figwidth()/10),2*r,facecolor=color,edgecolor=INK,lw=1.2,zorder=3))
    text(ax,x,y,s,15)

def save(fig,name):
    OUT.mkdir(parents=True,exist_ok=True)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    nodes=getattr(fig.axes[0], 'node_checks', [])
    for x,y,w,h,artist in nodes:
        extent=artist.get_window_extent(renderer)
        a=fig.axes[0].transData.transform((x,y))
        b=fig.axes[0].transData.transform((x+w,y+h))
        assert extent.x0 >= a[0]-2 and extent.x1 <= b[0]+2, artist.get_text()
        assert extent.y0 >= b[1]-2 and extent.y1 <= a[1]+2, artist.get_text()
    for i,(x,y,w,h,_) in enumerate(nodes):
        for xx,yy,ww,hh,other in nodes[i+1:]:
            assert not (min(x+w,xx+ww)-max(x,xx)>1 and
                        min(y+h,yy+hh)-max(y,yy)>1), other.get_text()
    for ext in ('png','svg'):
        fig.savefig(OUT/f'{name}.{ext}',facecolor='white')
    # This is the acceptance size, not a high-resolution zoom.
    fig.savefig(OUT/f'{name}_预览800.png',dpi=800/fig.get_figwidth(),facecolor='white')
    plt.close(fig)

def group(ax,x,y,w,h,label,color=GRAY):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=12',
        facecolor=color,edgecolor='#82909d',linewidth=1,linestyle='--',zorder=0))
    text(ax,x+15,y+22,label,13,ha='left',bold=True)


def overview():
    fig,ax=canvas(2080,'图1  TinyViM-S + UNet：常规层合框，21个编码block逐个保留')
    lx,rx=220,780
    def layer(cx,cy,label,color=BLUE,h=40,fs=13):
        box(ax,cx-175,cy-h/2,350,h,label,color,fs)
    def down(a,b): arrow(ax,lx,a,lx,b)
    layer(lx,100,'RGB：3×256×256\nImageNet均值、标准差归一化',GRAY,h=60,fs=12.5)
    down(130,152)
    layer(lx,185,'Conv 3×3, s=2：3→24 → BN → GELU\n24×128×128',BLUE,h=66,fs=12)
    down(218,237)
    layer(lx,270,'Conv 3×3, s=2：24→48 → BN → GELU\n48×64×64',BLUE,h=66,fs=12)
    prev=303
    specs=[
        (1,48,64,330,225,[385,445,505],['LocalBlock','LocalBlock','TinyViM Block'],550),
        (2,64,32,655,225,[710,770,830],['LocalBlock','LocalBlock','TinyViM Block'],875),
        (3,168,16,980,460,[1035+44*i for i in range(9)],
         ['LocalBlock']*4+['TinyViM Block']+['LocalBlock']*3+['TinyViM Block'],1435),
        (4,224,8,1540,355,[1595+50*i for i in range(6)],
         ['LocalBlock']*5+['TinyViM Block'],1890),
    ]
    global_id=0
    for si,c,hw,gy,gh,ys,kinds,end in specs:
        group(ax,25,gy,390,gh,(f'阶段{si} · {c}×{hw}×{hw}' if si<4 else '阶段4 · 瓶颈：224×8×8'),GRAY)
        for bi,(cy,kind) in enumerate(zip(ys,kinds)):
            global_id+=1
            if bi==0:
                down(prev,gy-4)
                if cy-17>gy+34: down(gy+34,cy-17)
            else: down(prev,cy-17)
            label=f'{global_id}  {kind}' if kind=='LocalBlock' else f'{global_id}  TinyViM Block / SS2D'
            layer(lx,cy,label,YELLOW if kind=='LocalBlock' else GREEN,h=34,fs=12.5)
            prev=cy+17
        down(prev,end); prev=end
        if si<4:
            nc=specs[si][1];cy=end+60
            down(prev,cy-25)
            layer(lx,cy,f'Conv 3×3, s=2：{c}→{nc} → BN',BLUE,h=50,fs=12.5)
            prev=cy+25
    down(prev,1921); layer(lx,1940,'BatchNorm：224',GRAY,h=38)
    route(ax,[(lx,1959),(lx,2000),(385,2000)])
    box(ax,385,1975,230,50,'1×1 Conv：224→512\n512×8×8',BLUE,12)
    route(ax,[(615,2000),(rx,2000),(rx,1727)])
    layer(rx,1700,'转置卷积 2×2, s=2\n512→256 · 256×16×16',BLUE,h=54,fs=12)
    arrow(ax,rx,1673,rx,1453)
    previous=None
    for i,(join,c,in_c,hw,skip_c,conv_y) in enumerate([
        (1435,256,512,16,168,1370),
        (875,128,256,32,64,810),
        (550,64,128,64,48,490),
        (None,32,32,128,None,325),
        (None,16,16,256,None,160),
    ]):
        if i in (1,2):
            up_y=(previous+join)/2
            arrow(ax,rx,previous,rx,up_y+27)
            layer(rx,up_y,f'转置卷积 2×2, s=2\n{c*2}→{c} · {c}×{hw}×{hw}',BLUE,h=54,fs=12)
            arrow(ax,rx,up_y-27,rx,join+18)
        elif i>=3:
            up_y=405 if i==3 else 240
            arrow(ax,rx,previous,rx,up_y+24)
            layer(rx,up_y,f'转置卷积 2×2, s=2：{c*2}→{c}',BLUE,h=48,fs=12)
            arrow(ax,rx,up_y-24,rx,conv_y+33)
        if join is not None:
            op(ax,rx,join,'C',r=16)
            arrow(ax,lx,join,rx-18,join)
            text(ax,500,join-25,f'1×1 Conv：{skip_c}→{c}',11.5)
            text(ax,500,join+23,f'{c}×{hw}×{hw}',11)
            arrow(ax,rx,join-18,rx,conv_y+33)
        layer(rx,conv_y,f'Conv 3×3：{in_c}→{c} → BN → ReLU\nConv 3×3：{c}→{c} → BN → ReLU\n输出：{c}×{hw}×{hw}',GREEN,h=66,fs=11.5)
        previous=conv_y-33
    arrow(ax,rx,previous,rx,110)
    layer(rx,80,'1×1 Conv：16→1 → Sigmoid\n病灶概率图：1×256×256',PURPLE,h=60,fs=12.5)
    text(ax,500,2060,'同一框内按文字顺序执行；C：通道拼接；BN：BatchNorm；s=2：步长2',11.5)
    save(fig,'TinyViM整网与全部block')


def local():
    fig,ax=canvas(660,'图3  LocalBlock：RepDW-3 → FFN → 缩放与残差',width=1200)
    box(ax,340,65,320,44,'输入特征 X',GRAY)
    arrow(ax,500,109,500,151)
    group(ax,110,135,780,240,'RepDW：重参数化深度卷积',BLUE)
    for x,label in [(150,'3×3深度卷积+BN'),(410,'1×1 深度卷积'),(670,'原特征直接传递')]:
        box(ax,x,197,180,60,label,GRAY,14)
        arrow(ax,x+90,177,x+90,197)
        path(ax,[(x+90,257),(x+90,283)])
    path(ax,[(500,151),(500,177)])
    path(ax,[(240,177),(760,177)])
    path(ax,[(240,283),(760,283)])
    arrow(ax,500,283,500,292)
    op(ax,500,310,'+')
    arrow(ax,500,328,500,344)
    box(ax,330,344,340,30,'归一化',BLUE,14)
    arrow(ax,500,374,500,419)
    group(ax,220,402,560,104,'FFN：在每个位置组合通道',YELLOW)
    box(ax,265,446,470,42,'1×1卷积+BN → GELU → 1×1卷积+BN',GRAY,14)
    arrow(ax,500,419,500,446)
    arrow(ax,500,488,500,537)
    box(ax,280,537,440,38,'逐通道缩放 + DropPath',BLUE,14)
    arrow(ax,500,575,500,594)
    op(ax,500,612,'+')
    route(ax,[(340,87),(55,87),(55,612),(481,612)])
    arrow(ax,519,612,700,612)
    box(ax,700,590,250,44,'输出特征',GRAY,16)
    save(fig,'LocalBlock与FFN内部结构')


def tinyvim():
    fig,ax=canvas(1475,'图2  TinyViM Block：当前代码展开（阶段1—3）',width=1200)
    box(ax,330,60,340,45,'输入特征 X',GRAY)
    arrow(ax,500,105,500,165)
    group(ax,60,132,890,1078,'Laplace Mixer · 在代码中由 SS2D 类实现',BLUE)
    box(ax,330,178,340,40,'1×1卷积 + BN：组合通道',GRAY,14)
    arrow(ax,500,165,500,178)
    arrow(ax,500,218,500,236)
    group(ax,95,236,810,166,'RepDW-7 · 两个方向各用三路结构（见图4）',GRAY)
    path(ax,[(500,236),(500,278)])
    path(ax,[(290,278),(710,278)])
    arrow(ax,290,278,290,300); arrow(ax,710,278,710,300)
    box(ax,115,300,350,70,'前半通道：7×1 深度卷积+BN\n与1×1深度卷积、原特征相加 → BN',BLUE,11.5)
    box(ax,535,300,350,70,'后半通道：1×7 深度卷积+BN\n与1×1深度卷积、原特征相加 → BN',BLUE,11.5)
    path(ax,[(290,370),(290,390),(710,390),(710,370)])
    arrow(ax,500,390,500,410)
    box(ax,330,410,340,40,'拼回通道 → SiLU',PURPLE,14)
    arrow(ax,500,450,500,480)
    box(ax,330,480,340,40,'代码：此处再按比例切分（见图6）',GRAY,13)
    path(ax,[(500,520),(500,540)])
    path(ax,[(240,540),(760,540)])
    arrow(ax,240,540,240,560); arrow(ax,760,540,760,560)
    box(ax,95,560,290,45,'扫描组：前面一部分通道',GRAY,13)
    box(ax,625,560,270,45,'局部卷积组：其余通道',GRAY,13)
    group(ax,80,614,510,114,'Laplace Decoupling：池化 + 差值',YELLOW)
    arrow(ax,240,605,240,650)
    box(ax,95,650,290,55,'平均池化 → 低分辨率特征\n送入下方 SS2D 核心',GREEN,12)
    arrow(ax,240,705,240,740)
    group(ax,85,740,475,256,'Mamba扫描核心：SS2D（见图5）',GREEN)
    text(ax,322,796,'逐行正向 →     ← 整条序列反向',13)
    text(ax,322,835,'逐列正向 ↓     ↑ 整条序列反向',13)
    text(ax,322,874,'每路沿顺序传递状态，',13)
    text(ax,322,901,'用当前特征更新状态并生成输出',13)
    box(ax,110,940,425,41,'四路还原位置并相加 → 归一化',GRAY,12.5)
    box(ax,395,650,185,72,'细节差值\n原特征 − 最近邻放大\n(池化结果)',GRAY,10.5)
    route(ax,[(385,582),(487,582),(487,650)])
    arrow(ax,385,680,395,680)
    arrow(ax,322,996,322,1018)
    text(ax,322,1039,'双线性插值 → 恢复原尺寸',13)
    arrow(ax,322,1060,322,1082)
    op(ax,322,1100,'+')
    route(ax,[(580,680),(597,680),(597,1100),(341,1100)])
    group(ax,625,640,270,335,'RepDW-3 · 每个通道各自卷积',GRAY)
    box(ax,650,695,220,43,'3×3 深度卷积 + BN',BLUE,12)
    box(ax,650,765,220,43,'1×1 深度卷积',BLUE,13)
    box(ax,650,835,220,43,'原特征直接传递',BLUE,13)
    path(ax,[(760,605),(915,605),(915,856)])
    for cy in [716,786,856]:
        arrow(ax,915,cy,870,cy)
        path(ax,[(650,cy),(614,cy)])
    path(ax,[(614,716),(614,907)])
    arrow(ax,614,907,742,907)
    op(ax,760,907,'+')
    arrow(ax,760,925,760,950)
    box(ax,670,950,180,30,'归一化',BLUE,12)
    route(ax,[(322,1118),(322,1139),(482,1139)])
    route(ax,[(760,980),(760,1139),(518,1139)])
    op(ax,500,1139,'C',r=14)
    text(ax,695,1179,'C：沿通道拼接',11,ha='left')
    arrow(ax,500,1156,500,1168)
    box(ax,330,1168,340,30,'1×1卷积 + BN',PURPLE,12)
    arrow(ax,500,1198,500,1215)
    op(ax,500,1233,'+')
    route(ax,[(330,82),(30,82),(30,1233),(481,1233)])
    arrow(ax,500,1251,500,1296)
    box(ax,260,1296,480,58,'FFN：在每个位置组合通道\n1×1卷积 → GELU → 1×1卷积',YELLOW,14)
    arrow(ax,500,1354,500,1384)
    op(ax,500,1402,'+')
    route(ax,[(519,1233),(925,1233),(925,1402),(519,1402)])
    arrow(ax,500,1420,500,1441)
    box(ax,330,1441,340,32,'输出特征 Y',GRAY,13)
    save(fig,'TinyViM完整block内部结构')


def scan_explained():
    fig,ax=canvas(1440,'图5  Mamba核心在此：SS2D四方向扫描与状态更新',width=1200)
    text(ax,140,83,'输入低分辨率特征',12,bold=True)
    for row in range(3):
        for col in range(3):
            x,y=70+col*45,125+row*45
            ax.add_patch(Rectangle((x,y),45,45,facecolor=BLUE,edgecolor=INK,lw=1))
            text(ax,x+22.5,y+22.5,str(1+row*3+col),15)
    text(ax,485,68,'CrossScan：四种排列',12,bold=True)
    text(ax,770,68,'四路扫描',11,bold=True)
    text(ax,915,68,'CrossMerge',11,bold=True)
    path(ax,[(205,192.5),(260,192.5)])
    path(ax,[(260,110),(260,350)])
    sequences=[('行正向','1  2  3  4  5  6  7  8  9'),
               ('行反向','9  8  7  6  5  4  3  2  1'),
               ('列正向','1  4  7  2  5  8  3  6  9'),
               ('列反向','9  6  3  8  5  2  7  4  1')]
    for cy,(label,seq) in zip([110,190,270,350],sequences):
        arrow(ax,260,cy,300,cy)
        box(ax,300,cy-25,390,50,label+'：'+seq,GRAY,12)
        arrow(ax,690,cy,710,cy)
        box(ax,710,cy-25,120,50,'见下方计算',GREEN,11)
        arrow(ax,830,cy,850,cy)
        box(ax,850,cy-25,130,50,'还原位置',BLUE,12)
        path(ax,[(980,cy),(990,cy)])
    route(ax,[(990,110),(990,408),(800,408),(800,430)])
    box(ax,635,430,330,50,'四路求和 → LayerNorm',PURPLE,12)
    text(ax,310,446,'整条序列传递状态；换行、换列不重置。',11)
    group(ax,35,530,930,855,'每个方向内部：先生成系数，再沿64个位置更新状态',YELLOW)
    box(ax,280,590,440,65,'第t个位置的输入特征\n第3阶段：每个位置84个通道数值',GRAY,13)
    arrow(ax,500,655,500,710)
    box(ax,130,710,740,75,'线性映射1：84 → 27\n生成三部分：步长草稿11维、写入系数B 8维、读出系数C 8维',GREEN,13)
    path(ax,[(500,785),(500,815)])
    path(ax,[(265,815),(745,815)])
    arrow(ax,265,815,265,860); arrow(ax,745,815,745,860)
    box(ax,60,860,410,100,'步长草稿 → 线性映射2：11→84\n加偏置，再经softplus变成正值\n得到该位置每个通道的 Δ[t]',GRAY,12.5)
    box(ax,550,860,390,100,'B[t]：该位置怎样写入状态\nC[t]：该位置怎样读出状态\n这两份系数由线性映射1直接产生',GRAY,12.5)
    arrow(ax,265,960,265,1055); arrow(ax,745,960,745,1055)
    box(ax,70,1055,860,145,'',GREEN,13)
    text(ax,500,1096,r'$h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$',25)
    text(ax,500,1160,r'$y_t = C_t h_t + D x_t$',25)
    text(ax,500,1242,r'$\bar{A}_t$、$\bar{B}_t$ 是离散化后的系数；A、D是跨位置共用的训练参数。',12)
    text(ax,500,1284,'Δ[t]意为“第t个位置的步长”；t是扫描位置1…64，不是epoch或训练迭代。',12)
    text(ax,500,1326,r'状态初始为0；每通道8个分量；$C_t h_t$表示对8个状态分量加权求和。',12)
    text(ax,500,1365,'当前扫描调用没有原始Mamba的z乘法门控；详细区别与原图见正文。',11.5)
    text(ax,500,1412,'上方3×3仅演示排列；第3阶段实际扫描84通道的8×8网格。',12)
    save(fig,'SS2D四方向与状态更新')


def repdw_expanded():
    fig,ax=canvas(1340,'图4  RepDW：三路内部结构与两种实际用法')
    text(ax,500,70,'DW：每个通道单独做卷积；BN：BatchNorm；C：沿通道拼接',12)
    # Every RepDW unit contains the same three paths; only the main kernel varies.
    def unit(x,y,w,kernel,label):
        group(ax,x,y,w,300,label,BLUE)
        cx=x+w/2
        arrow(ax,cx,y+40,cx,y+58)
        xs=[x+w*f for f in (.17,.5,.83)]
        path(ax,[(xs[0],y+58),(xs[-1],y+58)])
        labels=[f'{kernel} DW\nBN','1×1 DW','恒等传递\n原特征']
        bw=w*.29
        for xx,lab in zip(xs,labels):
            arrow(ax,xx,y+58,xx,y+80)
            box(ax,xx-bw/2,y+80,bw,65,lab,GRAY,12)
            path(ax,[(xx,y+145),(xx,y+170)])
        path(ax,[(xs[0],y+170),(xs[-1],y+170)])
        arrow(ax,cx,y+170,cx,y+183)
        op(ax,cx,y+200,'+')
        arrow(ax,cx,y+218,cx,y+240)
        box(ax,cx-65,y+240,130,36,'BN',PURPLE,13)
        return cx,y+276
    box(ax,330,105,340,45,'RepDW-7 输入：168×16×16',GRAY,14)
    path(ax,[(500,150),(500,180)])
    path(ax,[(255,180),(745,180)])
    arrow(ax,255,180,255,250); arrow(ax,745,180,745,250)
    x1,y1=unit(35,210,440,'7×1','前84通道：7×1方向')
    x2,y2=unit(525,210,440,'1×7','后84通道：1×7方向')
    route(ax,[(x1,y1),(x1,560),(482,560)])
    route(ax,[(x2,y2),(x2,560),(518,560)])
    op(ax,500,560,'C')
    arrow(ax,500,578,500,610)
    box(ax,330,610,340,42,'RepDW-7 输出：168×16×16',GREEN,13)
    arrow(ax,500,652,500,685)
    box(ax,390,685,220,40,'SiLU（后续激活）',PURPLE,13)
    text(ax,500,758,'两组通道并行处理；每组内部三路相加；两组之间用 C 拼接。',13)
    box(ax,330,810,340,42,'RepDW-3 输入：任意 C×H×W',GRAY,13)
    arrow(ax,500,852,500,900)
    _,bottom=unit(190,860,620,'3×3','所有输入通道：3×3邻域')
    arrow(ax,500,bottom,500,1190)
    box(ax,330,1190,340,42,'RepDW-3 输出：同一 C×H×W',GREEN,13)
    text(ax,500,1272,'显式 fuse() 后：每个三路单元可合并为一个同形状 DW。',12)
    text(ax,500,1306,'7×1 仍是7×1，1×7仍是1×7；两组不会变成一个7×7。',12)
    save(fig,'RepDW三路与方向分组展开')


def decoupling_expanded():
    fig,ax=canvas(1100,'图5  解耦与恢复：先得到 L，再保留差值 R，最后加回')
    group(ax,30,75,940,210,'① 平均池化：得到低分辨率特征 L',BLUE)
    for x,label in [(60,'A：84×16×16'),(365,'2×2平均池化\n步长2'),(720,'L：84×8×8')]:
        box(ax,x,165,220,64,label,GRAY,13)
    arrow(ax,280,197,365,197); arrow(ax,585,197,720,197)
    group(ax,30,320,940,300,'② 原特征减去低分辨率重建：保留差值 R',YELLOW)
    box(ax,60,420,220,60,'A：84×16×16',GRAY,13)
    box(ax,720,420,220,60,'L：84×8×8',GRAY,13)
    op(ax,500,450,'−')
    arrow(ax,280,450,482,450)
    arrow(ax,720,450,518,450)
    text(ax,620,411,'最近邻插值 ×2',12)
    arrow(ax,500,468,500,530)
    box(ax,335,530,330,55,'R = A − 最近邻放大(L)\n84×16×16',GRAY,12.5)
    group(ax,30,655,940,390,'③ L扫描后放大，再加回同尺寸的 R',GREEN)
    box(ax,60,820,190,65,'L\n84×8×8',GRAY,13)
    box(ax,315,820,255,65,'SS2D核心\n扫描 → 合并 → LN',BLUE,13)
    arrow(ax,250,852.5,315,852.5)
    op(ax,830,852.5,'+')
    arrow(ax,570,852.5,812,852.5)
    text(ax,688,814,'双线性插值 ×2',12)
    box(ax,730,720,200,55,'R：84×16×16',GRAY,12.5)
    arrow(ax,830,775,830,834.5)
    arrow(ax,830,870.5,830,960)
    box(ax,635,960,325,55,'扫描支输出\n84×16×16',GRAY,13)
    text(ax,320,988,'每行按编号读；同名 L、R 是同一份特征。',11.5)
    text(ax,500,1075,'示例为第3阶段；第4阶段直接扫描，跳过本图的池化、差值和插值。',12)
    save(fig,'Laplace解耦与细节恢复展开')


def channel_splits():
    fig,ax=canvas(980,'图6  代码中的两次切分：先分方向，再分扫描与局部处理')
    text(ax,500,80,'每条色带按通道顺序从左到右排列，每个通道保留完整的 H×W 空间。',12)
    text(ax,500,121,'第一次：RepDW-7固定对半 → 两组拼回 → SiLU → 第二次：按阶段比例切分',12)
    for i,(c,hw,scan) in enumerate([(48,64,12),(64,32,32),(168,16,84),(224,8,168)]):
        top=190+i*180
        text(ax,32,top+13,f'阶段{i+1}',13,ha='left',bold=True)
        text(ax,32,top+49,f'{c}×{hw}×{hw}',11,ha='left')
        x,w=170,790
        half=c//2
        box(ax,x,top,w/2,40,f'前{half}通道：7×1三路单元',BLUE,11.5)
        box(ax,x+w/2,top,w/2,40,f'后{half}通道：1×7三路单元',PURPLE,11.5)
        for xx in [x+w*.2,x+w*.8]: arrow(ax,xx,top+40,xx,top+72)
        ratio=scan/c
        box(ax,x,top+72,w*ratio,42,f'扫描：前{scan}通道',GREEN,11.5)
        box(ax,x+w*ratio,top+72,w*(1-ratio),42,f'局部：后{c-scan}通道',YELLOW,11.5)
        text(ax,560,top+143,f'扫描比例 α={ratio:g}；两组空间尺寸均为{hw}×{hw}',12)
    text(ax,500,930,'扫描组交给扫描核心；局部组交给RepDW-3。切分比例固定，没有通道排序或重排。',12)
    text(ax,500,963,'阶段1和4的两次切分边界不同；阶段2和3的边界恰好重合。',12)
    save(fig,'两次通道切分与四阶段比例')


def mamba_equations():
    fig,ax=canvas(540,'Mamba选择性状态空间：先看状态更新，再看系数来源')
    text(ax,500,110,r'$h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$',28)
    text(ax,500,200,r'$y_t = C_t h_t + D x_t$',28)
    text(ax,500,295,r'$B_t=s_B(x_t),\qquad C_t=s_C(x_t)$',23)
    text(ax,500,370,r'$\Delta_t=\mathrm{softplus}(s_{\Delta}(x_t))$',23)
    text(ax,500,443,'x：输入；h：状态；y：扫描输出；t：序列位置。',14)
    text(ax,500,484,'s表示学到的映射；横线表示离散化后的系数，来源见正文。',13)
    text(ax,500,519,'A、D可训练且跨位置共用；B、C、Δ由当前位置输入生成。',13)
    save(fig,'Mamba选择性状态更新公式')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/'references/code/TinyViM/fig/whole.png', OUT/'TinyViM官方原图.png')
    overview(); tinyvim(); local(); repdw_expanded(); scan_explained(); channel_splits(); mamba_equations()
if __name__=='__main__':
    main()

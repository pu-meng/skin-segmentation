"""Code-checked source backbone + proposed bridge, not an implemented network."""
from pathlib import Path
import ast
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parents[1]
FONT = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['svg.fonttype'] = 'path'
INK = '#28384A'
COL = {'base':'#f6e8a6', 'swin':'#E8F2FC', 'new':'#e6d5f7', 'gate':'#FFF0D8', 'out':'#cdefd8', 'context':'#cfe3f6'}

def start(w,h):
    global fig,ax
    fig,ax=plt.subplots(figsize=(14,15) if w == 17 else (12,12))
    fig.subplots_adjust(0,0,1,1)
    ax.set(xlim=(0,w),ylim=(0,h)); ax.axis('off')

def txt(x,y,s,size=12,color=INK):
    ax.text(x,y,s,fontproperties=FONT,fontsize=max(size,13),ha='center',va='center',color=color,linespacing=1.25,zorder=5)

def box(x,y,w,title,sub='',kind='base',h=.85):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=.02,rounding_size=.1',facecolor=COL[kind],edgecolor=INK,lw=1.3,zorder=3))
    txt(x,y+(.10 if h <= .65 else .15) if sub else y,title,17)
    if sub: txt(x,y+(-.16 if h <= .65 else -.20),sub,13)

def arr(a,b,color=INK,dashed=False):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,lw=1.4,color=color,linestyle='--' if dashed else '-',shrinkA=0,shrinkB=0,zorder=2))

def path(points,color=INK,dashed=False):
    # Keep directed endpoints; replace orthogonal detours with one flow curve.
    if len(points) == 2:
        arr(points[0], points[-1], color, dashed)
        return
    a, b = points[0], points[-1]
    c1 = points[1]
    c2 = points[-2]
    curve = MplPath([a,c1,c2,b], [MplPath.MOVETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4])
    ax.add_patch(FancyArrowPatch(path=curve,arrowstyle='-|>',mutation_scale=12,lw=1.4,color=color,linestyle='--' if dashed else '-',zorder=2))

def plus(x,y):
    ax.add_patch(Circle((x,y),.18,facecolor='white',edgecolor=INK,lw=1.3,zorder=3)); txt(x,y,'+',17)

def save(name):
    out=ROOT/'figures'; out.mkdir(exist_ok=True)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for t in ax.texts:
        b=t.get_window_extent(renderer)
        if b.x0<0 or b.y0<0 or b.x1>fig.bbox.x1 or b.y1>fig.bbox.y1:
            raise RuntimeError('Text outside canvas: '+t.get_text())
    fig.savefig(out/(name+'.svg'),facecolor='white',metadata={'Date':None})
    fig.savefig(out/(name+'.png'),dpi=300,facecolor='white')
    # Previews are rendered from the actual SVG during verification.
    ET.parse(out/(name+'.svg')); plt.close(fig)
    print(name)

start(17,19)
txt(8.5,18.5,'01(a)｜在完整 MedNeXt 主模型上增加一处桥接',22)
txt(8.5,17.98,'设计图：主干据真实 forward 核对；紫色桥接模块尚待实现；形状示例使用 128³ CT patch',12)
box(6,17.2,2.8,'CT patch','[B,1,128,128,128]','swin')
box(6,15.85,2.8,'Stem','1 → 32','base',.65)
arr((6,16.775),(6,16.175))
path([(4.6,17.2),(2,17.2),(2,14.825)])
arr((6,15.525),(6,14.825))
ys=[14.4,12.2,10,7.8]
counts=[3,4,8,8]
for i,y in enumerate(ys):
    c=32*2**i; size=128//2**i
    box(6,y,2.8,f'Encoder Stage {i}',f'{counts[i]} blocks ｜ {c} ch, {size}³')
    if i<3:
        box(6,y-1.1,2.2,f'DownBlock {i}','stride 2','base',.6)
        arr((6,y-.425),(6,y-.8)); arr((6,y-1.4),(6,ys[i+1]+.425))

box(6,6.7,2.2,'DownBlock 3','256 → 512','base',.6)
arr((6,7.375),(6,7.0))
box(6,5.55,2.8,'Conv bottleneck','8 blocks ｜ 512 ch, 8³')
arr((6,6.4),(6,5.975))
box(6,4.12,3.8,'原模型的上下文模块','原 MHA＋MoE','context',.9)
arr((6,5.125),(6,4.57))
box(12.65,5.55,2.8,'UpBlock 3','512 → 256, 16³')
arr((7.9,4.12),(11.25,5.55))

box(2,14.4,3.1,'Swin Patch embedding','48 ch, 64³','swin')
box(2,12.2,3.1,'Swin Stage 1','2 blocks；96 ch, 32³','swin')
box(2,10,3.1,'Swin Stage 2','2 blocks；192 ch, 16³','swin')
arr((2,13.975),(2,12.625)); arr((2,11.775),(2,10.425))
box(2,8.7,3.1,'归一化＋encoder4','S = [B,192,16,16,16]','swin')
arr((2,9.575),(2,9.125))
box(10.35,7.8,3.7,'新增桥接模块 ×1','E、S、U → E′','new',1.1)
arr((7.4,7.8),(8.5,7.8))
path([(3.55,8.5),(4,8.3),(7.8,8.3),(8.5,8.1)],'#2f6fa3')
txt(2,6.95,'完整 Swin 还有 Stage 3、4\n本次只取 1/8 尺度，\n后续 stage 和 Swin decoder 不参与',11)
txt(10.0,8.65,'投影×1；专家×2；路由器×1',11)

for i in [3,2,1,0]:
    y=ys[i]; c=32*2**i; size=128//2**i
    plus(12.65,y)
    box(14.7,y,2.8,f'Decoder Stage {i}',f'{counts[i]} blocks ｜ {c} ch, {size}³')
    arr((12.83,y),(13.3,y))
    if i==3:
        arr((12.2,y),(12.47,y))
        arr((12.65,5.975),(12.65,y-.18))
        arr((11.25,5.55),(10.35,7.25))
        txt(14.6,6.65,'U：256 ch, 16³',10)
    else:
        arr((7.4,y),(12.47,y))
        txt(10,y+.3,f'原 skip {i}',10)
        upy=y-1.1
        box(12.65,upy,2.2,f'UpBlock {i}','stride 2','base',.6)
        arr((14.7,ys[i+1]+.425),(12.65,upy-.3))
        arr((12.65,upy+.3),(12.65,y-.18))
box(14.7,16,2.8,'分割输出头','背景 / 肝脏 / 肿瘤','out')
arr((14.7,14.825),(14.7,15.575))
txt(10.4,15.8,'冻结：两个已训练的源网络\n训练：紫色桥接模块内部参数',12)
txt(8.5,1.65,'仅在 skip 3 加桥接；其余连接保留',14)
txt(8.5,.92,'同一 CT、同一空间坐标；Swin 取至 Stage 2，最终使用主解码器输出',12)
save('01a_完整网络与插入位置')

start(15,15)
txt(7.5,14.45,'01(b)｜一处桥接模块内部到底增加了什么',21)
txt(7.5,13.85,'E、U：256 通道；S：192 通道；本图采用 16×16×16 特征网格',12)
box(3.3,12.95,3.3,'MedNeXt 特征 E','[B,256,16,16,16]','swin')
box(9.3,12.95,3.3,'Swin 特征 S','[B,192,16,16,16]','swin')
box(9.3,11.5,3.3,'空间对齐 + 通道投影','Conv3d 192 → 256, k=1','new')
arr((9.3,12.525),(9.3,11.925))
box(3.3,9.95,3.3,'残差专家 1','256 → 64 → 256','new')
box(9.3,9.95,3.3,'残差专家 2','256 → 64 → 256','new')
arr((3.3,12.525),(3.3,10.375)); arr((9.3,11.075),(9.3,10.375))
txt(6.3,9.08,'两专家：1×1×1 → GELU → 1×1×1\n末层权重、bias 零初始化',11)
box(3.3,8.15,3.3,'R1','[B,256,16,16,16]','new')
box(9.3,8.15,3.3,'R2','[B,256,16,16,16]','new')
arr((3.3,9.525),(3.3,8.575)); arr((9.3,9.525),(9.3,8.575))
box(6.3,6.62,6.8,'按两路权重加权求和','w1 × R1 + w2 × R2','new')
arr((3.3,7.725),(4.3,7.045))
arr((9.3,7.725),(8.3,7.045))
plus(6.3,5.35)
arr((6.3,6.195),(6.3,5.53))
residual = MplPath([(1.65,12.95),(.4,12.95),(.4,8),(.6,6.1),(.7,5.35),(3,5.35),(6.12,5.35)], [MplPath.MOVETO]+[MplPath.CURVE4]*6)
ax.add_patch(FancyArrowPatch(path=residual,arrowstyle='-|>',mutation_scale=12,lw=1.4,color=INK,zorder=2))
txt(2.45,5.65,'原始 E 直接加回',11)
plus(6.3,4.05); arr((6.3,5.17),(6.3,4.23))
box(9.4,4.05,3.3,'MedNeXt 特征 U','[B,256,16,16,16]','swin')
arr((7.75,4.05),(6.48,4.05))
box(6.3,2.62,6.8,'输出 E′ + U → dec_block_3','[B,256,16,16,16]','out')
arr((6.3,3.87),(6.3,3.045))

box(12.85,12.05,3.1,'拼接 E、S′、U','[B,768,16,16,16]','gate')
box(12.85,10.3,3.1,'权重生成模块','768→64→2；GELU','gate')
arr((12.85,11.625),(12.85,10.725))
box(12.85,8.55,3.1,'两路原始分数','[B,2,16,16,16]','gate')
arr((12.85,9.875),(12.85,8.975))
box(12.85,6.8,3.1,'softmax(dim=1)','w1、w2：[B,1,16³]','gate')
arr((12.85,8.125),(12.85,7.225))
path([(11.3,6.8),(10.35,6.8),(10.35,6.62),(9.7,6.62)],'#A66B10',True)
txt(12.85,5.55,'每个位置两权重相加为1\n同一位置的256通道共用\n分数不等于肿瘤预测',11)
txt(7.5,1.43,'新增可训练模块共4个：通道投影 ×1、残差专家 ×2、权重生成模块 ×1',13)
txt(7.5,.72,'零初始化的是新增专家末层；原 LiTS 权重不清零。初始 R1=R2=0，E′=E。',12)
save('01b_桥接内部与张量尺寸')

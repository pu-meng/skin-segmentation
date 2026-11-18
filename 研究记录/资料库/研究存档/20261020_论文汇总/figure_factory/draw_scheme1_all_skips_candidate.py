"""Additional design figures. Keeps original scheme-1 figures unchanged."""
from pathlib import Path
import hashlib
import re
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[1]
FONT = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['svg.fonttype'] = 'path'
INK = '#28384a'
COL = {'base':'#f6e8a6','swin':'#e8f2fc','bridge':'#e6d5f7','freeze':'#e7e9ed','tune':'#fff0d8','out':'#cdefd8'}

def start(height):
    global fig,ax
    fig,ax=plt.subplots(figsize=(16,height))
    fig.subplots_adjust(0,0,1,1)
    ax.set(xlim=(0,16),ylim=(height,0)); ax.axis('off')

def text(x,y,s,size=18,color=INK):
    replacements = {'E′_i': r"$E_i^{\prime}$", 'S′_i': r"$S_i^{\prime}$",
                    'b_i1': r"$b_{i1}$", 'b_i2': r"$b_{i2}$", 'α_i': r"$\alpha_i$",
                    'E_i': r"$E_i$", 'S_i': r"$S_i$", 'U_i': r"$U_i$", 'R_i': r"$R_i$",
                    'C_i': r"$C_i$", 'D_i': r"$D_i$", 'H_i': r"$H_i$", 'W_i': r"$W_i$"}
    s = re.sub("|".join(re.escape(k) for k in replacements), lambda match: replacements[match.group()], s)
    s = re.sub(r"([ESU])([₀₁₂₃])", lambda m: "$" + m[1] + "_" + str("₀₁₂₃".index(m[2])) + "$", s)
    ax.text(x,y,s,fontproperties=FONT,fontsize=size,ha='center',va='center',color=color,linespacing=1.35,zorder=5)

def box(x,y,w,h,title,sub='',kind='base'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015,rounding_size=0.10',facecolor=COL[kind],edgecolor=INK,lw=1.4,zorder=3))
    text(x+w/2,y+h*(.32 if sub else .5),title,19)
    if sub: text(x+w/2,y+h*.71,sub,17)

def arrow(a,b):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=16,lw=1.5,color=INK,shrinkA=0,shrinkB=0,zorder=2))

def path(points):
    for a,b in zip(points[:-2],points[1:-1]):
        ax.plot([a[0],b[0]],[a[1],b[1]],color=INK,lw=1.5,zorder=1)
    arrow(points[-2],points[-1])

def plus(x,y):
    ax.add_patch(Circle((x,y),.20,facecolor='white',edgecolor=INK,lw=1.5,zorder=3));text(x,y,'+',22)

def save(name):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for t in ax.texts:
        b=t.get_window_extent(renderer)
        if b.x0 < 0 or b.y0 < 0 or b.x1 > fig.bbox.x1 or b.y1 > fig.bbox.y1:
            raise RuntimeError('Text outside canvas: '+t.get_text())
    target=ROOT/'figures'/f'{name}.svg'
    fig.savefig(target,facecolor='white',metadata={'Date':None})
    ET.parse(target)
    fig.savefig('/tmp/'+name+'.png',dpi=110,facecolor='white')
    plt.close(fig)
    print(target)


def overview():
    start(23)
    text(8,.42,'图13  方案1四层候选：桥接插在原 skip 的加号之前',25)
    text(8,.97,'按原图1的整网布局重画；主干连接保留，紫色为新增桥接。尚未实现／验证。',18)
    box(4.0,1.5,3.2,.9,'同一 CT patch','[B,1,128,128,128]','swin')
    path([(4,1.95),(2.1,1.95),(2.1,2.75)])
    box(.5,2.75,3.2,.85,'Swin辅助编码器',kind='swin')
    arrow((5.6,2.4),(5.6,2.75))
    box(4,2.75,3.2,.85,'MedNeXt输入投影',kind='base')
    text(9.3,2.55,'每层接法：原skip → 新桥接 → 原加号',18)
    text(14.05,2.55,'原解码器：由下往上',18)
    ys=[4.0,7.5,11.0,14.5]
    arrow((5.6,3.6),(5.6,4.0))
    arrow((2.1,3.6),(2.1,4.0))
    for level,y in enumerate(ys):
        c=32*2**level; n=128//2**level
        box(.5,y,3.2,1.0,'Swin对应尺度特征',f'S{chr(0x2080+level)}；位置待核对','swin')
        if level<3: arrow((2.1,y+1.0),(2.1,ys[level+1]))
        box(4,y,3.2,1.0,f'Encoder Stage {level}',f'E{chr(0x2080+level)}：[B,{c},{n},{n},{n}]','base')
        box(4.45,y+1.65,2.3,.75,f'DownBlock {level}',kind='base')
        arrow((5.6,y+1.0),(5.6,y+1.65))
        if level<3: arrow((5.6,y+2.4),(5.6,ys[level+1]))
        box(8.1,y,2.9,1.0,f'新增桥接 {level}',f'读取 E{chr(0x2080+level)}、S{chr(0x2080+level)}、U{chr(0x2080+level)}','bridge')
        arrow((7.2,y+.5),(8.1,y+.5))
        text(7.65,y+.15,f'skip {level}',16)
        path([(2.1,y+1),(2.1,y+1.25),(8.6,y+1.25),(8.6,y+1.0)])
        plus(11.85,y+.5)
        arrow((11,y+.5),(11.65,y+.5))
        box(12.85,y,2.85,1.0,f'Decoder Stage {level}',f'{c}通道，{n}³','base')
        arrow((12.05,y+.5),(12.85,y+.5))
        box(13.1,y+1.65,2.35,.75,f'UpBlock {level}',kind='base')
        path([(14.275,y+1.65),(14.275,y+1.35),(11.85,y+1.35),(11.85,y+.7)])
        path([(13.1,y+2.025),(10.25,y+2.025),(10.25,y+1.0)])
        text(11.6,y+2.35,f'U{chr(0x2080+level)}也交给桥接',16)
        if level>0: arrow((14.275,y),(14.275,ys[level-1]+2.4))
    arrow((5.6,16.9),(5.6,17.55))
    box(3.8,17.55,3.6,1.0,'原卷积瓶颈','512通道，8³','base')
    arrow((5.6,18.55),(5.6,19.15))
    box(3.8,19.15,3.6,1.0,'原 MHA＋MoE','主模型已有模块','base')
    path([(5.6,20.15),(5.6,20.7),(14.275,20.7),(14.275,16.9)])
    arrow((14.275,4.0),(14.275,3.65))
    box(12.85,2.95,2.85,.7,'原分割输出头',kind='out')
    box(.7,21.35,7.0,1.0,'LiTS：先学好四层桥接','双编码器与原主干固定；训练新增模块','bridge')
    box(8.35,21.35,7.0,1.0,'HCC：每层只调3个数','2个路由偏置＋1个修正强度，共12个','tune')
    text(8,22.72,'四条紫色桥接都位于编码器通向解码器的横向 skip 上；均不插在加号之后或解码块内部。',18)
    save('图13_方案1四层skip少参数适配候选')


def detail():
    start(22)
    text(8,.5,'图14  单层桥接展开：冻结信息处理，只调选择与力度',25)
    text(8,1.12,'这是图13中桥接 i 的内部；i = 0、1、2、3。两路专家都计算，不是稀疏计算。',18)
    box(1.0,1.8,6.3,1.05,'阶段1：LiTS源域训练','学好对齐、两专家和空间路由器','bridge')
    box(8.7,1.8,6.3,1.05,'阶段2：HCC适配／本图颜色','灰色＝冻结；橙色＝可训练','tune')
    arrow((7.3,2.325),(8.7,2.325))
    box(1.5,3.55,12,1.45,'同一层的三个输入：E_i、S_i、U_i','S_i先对齐得到S′_i；三者均为 [B,C_i,D_i,H_i,W_i]','swin')
    path([(5.0,5.0),(5.0,5.5),(4.0,5.5),(4.0,6.1)])
    path([(11.0,5.0),(11.0,6.1)])
    box(1.15,6.1,5.7,2.0,'两个独立残差专家（冻结）','E_i → 降维 → GELU → 升维 → R1\nS′_i → 降维 → GELU → 升维 → R2','freeze')
    box(8.3,6.1,5.7,2.0,'空间路由器主体（冻结）','读取 E_i、S′_i、U_i\n输出每个位置的两路原始分数 g1、g2','freeze')
    arrow((11.15,8.1),(11.15,8.8))
    box(8.3,8.8,5.7,1.45,'HCC学习：偏置 b_i1、b_i2','原始分数分别加偏置 → softmax','tune')
    arrow((11.15,10.25),(11.15,10.85))
    box(8.3,10.85,5.7,1.0,'位置权重 w1、w2','每个位置 w1＋w2＝1','out')
    box(4.6,12.65,6.8,1.15,'两路修正加权相加','R_i = w1 × R1 + w2 × R2','bridge')
    path([(4.0,8.1),(4.0,13.225),(4.6,13.225)])
    path([(11.15,11.85),(11.15,12.25),(9.5,12.25),(9.5,12.65)])
    arrow((8.0,13.8),(8.0,14.45))
    box(4.6,14.45,6.8,1.15,'HCC学习：修正强度 α_i','得到 α_i × R_i；初始化 α_i＝1','tune')
    arrow((8.0,15.6),(8.0,16.30))
    plus(8,16.5)
    path([(1.5,4.25),(.45,4.25),(.45,16.5),(7.8,16.5)])
    text(3.6,16.2,'原特征 E_i 直接保留',18)
    arrow((8,16.7),(8,17.25))
    box(4.6,17.25,6.8,1.0,'桥接输出 E′_i','E′_i = E_i + α_i × R_i','out')
    arrow((8,18.25),(8,18.95))
    plus(8,19.15)
    path([(13.5,4.25),(15.55,4.25),(15.55,19.15),(8.2,19.15)])
    text(12.1,18.83,'原解码器特征 U_i',18)
    arrow((8,19.35),(8,19.9))
    box(4.6,19.9,6.8,.8,'送入原 Decoder Stage i',kind='base')
    text(8,21.25,'HCC：b初始化0、α初始化1，起点保持源域桥接输出；四层共12个数（有效自由度更少）。',18)
    save('图14_方案1单层桥接与12参数候选')

if __name__ == '__main__':
    protected=[ROOT/'figures/图1_方案1完整网络与桥接位置.svg',ROOT/'figures/图2_方案1新增桥接模块内部.svg']
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    overview();detail()
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in before.items())
    print('Original figure 1 and figure 2 unchanged')

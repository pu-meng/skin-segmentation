"""Candidate convolution-gated replacement of Mamba steps 4/5; no model implementation."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '研究记录/figures'
OUT.mkdir(exist_ok=True)
FONT = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'svg.fonttype':'path','mathtext.fontset':'dejavusans','font.size':13})
INK='#304254'; BLUE='#e8f2fc'; PURPLE='#eee5f8'; ORANGE='#fff0d9'; GREEN='#dff3e7'
TEXTS=[]

def canvas(h):
    fig,ax=plt.subplots(figsize=(8,h/100),dpi=100)
    fig.subplots_adjust(0,0,1,1)
    ax.set(xlim=(0,800),ylim=(h,0)); ax.axis('off')
    return fig,ax

def txt(ax,x,y,s,size=13,color=INK):
    t=ax.text(x,y,s,ha='center',va='center',fontsize=size,fontproperties=FONT,color=color,linespacing=1.5)
    TEXTS.append(t)
    return t

def box(ax,x,y,w,h,s,color=BLUE,size=13):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0,rounding_size=7',facecolor=color,edgecolor=INK,linewidth=1.35,zorder=3))
    t=txt(ax,x,y,s,size);t.set_zorder(4)
    t._diagram_box = (x-w/2, y-h/2, x+w/2, y+h/2)
    return t

def arrow(ax,points,dashed=False):
    for a,b in zip(points[:-2],points[1:-1]):
        ax.plot([a[0],b[0]],[a[1],b[1]],color=INK,lw=1.4,ls='--' if dashed else '-',zorder=1)
    ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=11,lw=1.4,color=INK,linestyle='--' if dashed else '-',shrinkA=0,shrinkB=0,zorder=2))

def op(ax,x,y,s):
    ax.add_patch(Circle((x,y),15,facecolor='white',edgecolor=INK,lw=1.4,zorder=3))
    t=txt(ax,x,y,s,17); t.set_zorder(4)

def save(fig,name):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for t in TEXTS:
        if t.figure is fig:
            bb=t.get_window_extent(renderer)
            assert bb.x0>=0 and bb.x1<=800 and bb.y0>=0 and bb.y1<=fig.bbox.height,(name,t.get_text(),bb)
    for t in TEXTS:
        if t.figure is fig and hasattr(t, '_diagram_box'):
            bb=t.get_window_extent(renderer)
            corners=t.axes.transData.inverted().transform([[bb.x0,bb.y0],[bb.x1,bb.y1]])
            left,top,right,bottom=t._diagram_box
            assert corners[:,0].min()>=left+3 and corners[:,0].max()<=right-3, (name,t.get_text(),'horizontal overflow')
            assert corners[:,1].min()>=top+3 and corners[:,1].max()<=bottom-3, (name,t.get_text(),'vertical overflow')
    fig.savefig(OUT/f'{name}.svg',facecolor='white')
    fig.savefig(OUT/f'{name}.png',dpi=200,facecolor='white')
    fig.savefig(OUT/f'{name}_preview800.png',dpi=100,facecolor='white')
    plt.close(fig)



f,a=canvas(1800)
txt(a,400,22,'第三版候选 · 卷积内容 + 内层调节 + 外层门控',14)
txt(a,400,51,r'$L=HW,\quad I=EC\quad$（只替换原第4、5步）',12)
box(a,355,110,500,70,'图像输入 → 展平并转置\n'+r'$[B,C,H,W]\to X:[B,L,C]$',BLUE,13)
box(a,355,215,500,90,'1. 保留输入投影与拆分\n'+r'$(F,Z)=\mathrm{Split}(\mathrm{Linear}(\mathrm{LN}(X)))$'+'\n'+r'$F,Z:[B,L,I]$',PURPLE,12)
arrow(a,[(355,145),(355,170)])
box(a,355,330,400,90,'2. 保留原因果卷积与 SiLU\n'+r'$T=\mathrm{SiLU}(\mathrm{DWConv1d}(F))$'+'\n'+r'$T:[B,L,I]$',PURPLE,12)
arrow(a,[(355,260),(355,285)])
box(a,690,330,150,90,'外层门控\n'+r'$\mathrm{SiLU}(Z)$'+'\n'+r'$[B,L,I]$',BLUE,12)
arrow(a,[(605,215),(690,215),(690,285)])
box(a,235,490,400,150,'2. 保留系数生成\n'+r'$Q=\mathrm{Linear}(T)$'+'\n'+r'$(\rho,B_s,C_s)=\mathrm{Split}(Q;R,N,N)$'+'\n'+r'$\rho:[B,L,R]\quad B_s,C_s:[B,L,N]$',PURPLE,12)
arrow(a,[(295,375),(295,395),(235,395),(235,415)])
box(a,235,650,400,100,'3. 保留步长生成\n'+r'$\Delta=\mathrm{softplus}(\mathrm{Linear}(\rho))$'+'\n'+r'$\Delta:[B,L,I]$',PURPLE,12)
arrow(a,[(235,565),(235,600)])
box(a,575,625,240,420,'4. 新增二维卷积分支\n'+
    r'$T\to[B,I,H,W]$'+'\n'+
    r'$\mathrm{DWConv2d}(k\times k)$'+'\n'+
    r'$\to\mathrm{GroupNorm}$'+'\n'+
    r'$\to\mathrm{PWConv}(I,rI)$'+'\n'+
    r'$\to\mathrm{GELU}$'+'\n'+
    r'$\to\mathrm{PWConv}(rI,I)$'+'\n'+
    r'$\to U:[B,L,I]$'+'\n'+
    '本图不另加块内残差',ORANGE,11)
arrow(a,[(455,375),(455,395),(575,395),(575,415)])
box(a,235,830,400,240,'4—5. 内层调节（不再递推状态）\n'+
    r'$A=-\exp(A_{\log})\quad A:[I,N]$'+'\n'+
    r'$G_U=\sum_{n=1}^N C_{s,n}\exp(\Delta A_n)$'+'\n'+
    r'$G_T=\Delta\sum_{n=1}^N C_{s,n}B_{s,n}$'+'\n'+
    r'$G_U,G_T:[B,L,I]$',ORANGE,12)
arrow(a,[(235,700),(235,710)])
box(a,355,1030,500,100,'5. 用内层系数调节两路内容\n'+r'$Y_0=G_U\odot U+G_T\odot T$'+'\n'+r'$Y_0:[B,L,I]$',ORANGE,13)
arrow(a,[(235,950),(235,980)])
arrow(a,[(575,835),(575,980)])
arrow(a,[(155,330),(15,330),(15,1030),(105,1030)])
txt(a,35,983,r'$T$',13)
box(a,355,1160,500,90,'5. 保留 D 内容直连\n'+r'$Y_1=Y_0+D_{skip}\odot T$'+'\n'+r'$D_{skip}:[I]\quad Y_1:[B,L,I]$',ORANGE,12)
arrow(a,[(355,1080),(355,1115)])
arrow(a,[(15,1030),(15,1160),(105,1160)])
box(a,355,1280,500,90,'6. 保留外层门控\n'+r'$Y_2=Y_1\odot\mathrm{SiLU}(Z)$'+'\n'+r'$Y_2:[B,L,I]$',PURPLE,13)
arrow(a,[(355,1205),(355,1235)])
arrow(a,[(690,375),(760,375),(760,1280),(605,1280)])
box(a,355,1400,500,90,'7. 保留输出投影与外层残差\n'+r'$Y=X+\mathrm{Linear}(Y_2)$'+'\n'+r'$I\to C\quad Y:[B,L,C]$',GREEN,13)
arrow(a,[(355,1325),(355,1355)])
arrow(a,[(605,110),(785,110),(785,1400),(605,1400)])
box(a,355,1505,500,60,'转置并还原\n'+r'$[B,L,C]\to[B,C,H,W]$',GREEN,13)
arrow(a,[(355,1445),(355,1475)])
box(a,400,1665,740,190,'同一图像输入下的乘加主项变化（新版 − 原版）\n'+
    r'$\Delta F\approx BHW\left[4rI^2+2Ik^2-4I(N-1)+(2N-1)\right]$'+'\n'+
    r'$I=EC$'+'\n'+
    r'$F_T\approx24BHWC^2+4B(HW)^2C$',ORANGE,13)
txt(a,400,1780,'候选设计，尚未实现或训练；乘加口径与原图一致，计数范围见正文。',10)
save(f,'07_卷积双层门控候选与计算量')
print(OUT/'07_卷积双层门控候选与计算量.svg')

"""Transformer image/token shape mapping and forward FLOPs from comparison_mixers/models.py."""
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



f,a=canvas(1840)
txt(a,400,24,'Transformer Block · 图像到序列 · 逐项 FLOPs',15)
txt(a,400,54,r'$L=HW,\quad D=C,\quad a=4,\quad d_h=C/a$',13)
box(a,355,115,500,66,'输入特征图\n'+r'$X:[B,C,H,W]$',BLUE,13)
box(a,355,215,500,80,'1. 通道 LayerNorm（ChannelNorm）\n'+r'$[B,C,H,W]\to[B,H,W,C]\to[B,C,H,W]$',PURPLE,12)
arrow(a,[(355,148),(355,175)])
box(a,355,335,500,100,'2. 展平空间，再交换通道与序列轴\n'+r'$[B,C,H,W]\to[B,C,HW]\to[B,HW,C]$'+'\n'+r'$z:[B,L,D]\quad L=HW,\ D=C$',BLUE,12)
arrow(a,[(355,255),(355,285)])
box(a,355,470,500,110,'3. QKV 线性投影\n'+r'$[B,L,D]\to[B,L,3D]$'+'\n'+r'$F_{QKV}\approx6BLD^2=6B(HW)C^2$',PURPLE,13)
arrow(a,[(355,385),(355,415)])
box(a,355,610,500,110,'4. 拆成多头（a=4）\n'+r'$Q,K,V:[B,a,L,d_h]$'+'\n'+r'$d_h=D/a=C/a$',BLUE,13)
arrow(a,[(355,525),(355,555)])
box(a,355,770,500,150,'5. 注意力分数\n'+r'$S=\mathrm{softmax}(QK^\top/\sqrt{d_h})$'+'\n'+r'$S:[B,a,L,L]=[B,a,HW,HW]$'+'\n'+r'$F_{QK}\approx2BL^2D=2B(HW)^2C$',ORANGE,13)
arrow(a,[(355,665),(355,695)])
box(a,355,935,500,130,'6. 注意力加权\n'+r'$U=SV\quad U:[B,a,L,d_h]$'+'\n'+r'$F_{SV}\approx2BL^2D=2B(HW)^2C$',ORANGE,13)
arrow(a,[(355,845),(355,870)])
box(a,355,1090,500,130,'7. 合并多头与输出投影\n'+r'$[B,a,L,d_h]\to[B,L,D]\to[B,L,D]$'+'\n'+r'$F_O\approx2BLD^2=2B(HW)C^2$',PURPLE,12)
arrow(a,[(355,1000),(355,1025)])
box(a,355,1235,500,110,'8. 还原特征图，再加第一条残差\n'+r'$[B,L,D]\to[B,D,L]\to[B,C,H,W]$'+'\n'+r'$X_1=X+\mathrm{Attention}(\mathrm{LN}_1(X))$',GREEN,12)
arrow(a,[(355,1155),(355,1180)])
arrow(a,[(605,115),(755,115),(755,1235),(605,1235)])
txt(a,725,1198,r'$X$',13)
box(a,355,1400,500,170,'9. 第二次 LayerNorm 与 FFN（1×1卷积）\n'+r'$[B,C,H,W]\to[B,4C,H,W]\to[B,C,H,W]$'+'\n'+r'$\mathrm{Conv}(C,4C)\to\mathrm{GELU}\to\mathrm{Conv}(4C,C)$'+'\n'+r'$F_{FFN}\approx8B(HW)C^2+8B(HW)C^2$'+'\n'+r'$=16B(HW)C^2=16BLD^2$',PURPLE,12)
arrow(a,[(355,1290),(355,1315)])
box(a,355,1550,500,80,'10. 第二条残差\n'+r'$Y=X_1+\mathrm{FFN}(\mathrm{LN}_2(X_1))$'+'\n'+r'$Y:[B,C,H,W]$',GREEN,12)
arrow(a,[(355,1485),(355,1510)])
arrow(a,[(605,1235),(690,1235),(690,1550),(605,1550)])
txt(a,662,1515,r'$X_1$',13)
box(a,400,1710,740,180,'完整 Block 的主项 FLOPs\n'+
    r'$F_T\approx(6+2+16)BLD^2+(2+2)BL^2D$'+'\n'+
    r'$=24BLD^2+4BL^2D$'+'\n'+
    r'$=24B(HW)C^2+4B(HW)^2C$',ORANGE,15)
txt(a,400,1820,'1次乘法 + 1次加法 = 2 FLOPs；省略偏置、归一化、激活、Softmax与残差加法。',10)
save(f,'06_Transformer图像序列转换与FLOPs')
print(OUT/'06_Transformer图像序列转换与FLOPs.svg')

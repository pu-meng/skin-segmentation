"""Dimension-free candidate with B/C coefficient branches removed; A keeps its N components."""
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



f,a=canvas(1300)
txt(a,400,25,'第三版候选 · 去除 B、C 分支',15)
box(a,355,85,500,60,'输入特征图 → 展平并转置\n'+r'$X$',BLUE,13)
box(a,355,180,500,70,'1. 输入投影与拆分\n'+r'$(F,Z)=\mathrm{Split}(\mathrm{Linear}(\mathrm{LN}(X)))$',PURPLE,12)
arrow(a,[(355,115),(355,145)])
box(a,355,280,400,70,'2. 因果深度卷积与 SiLU\n'+r'$T=\mathrm{SiLU}(\mathrm{DWConv1d}(F))$',PURPLE,12)
arrow(a,[(355,215),(355,245)])
box(a,690,280,150,70,'外层门控\n'+r'$\mathrm{SiLU}(Z)$',BLUE,13)
arrow(a,[(605,180),(690,180),(690,245)])
box(a,235,415,400,80,'2. 步长中间特征\n'+r'$\rho=\mathrm{Linear}(T)$',PURPLE,13)
arrow(a,[(295,315),(295,338),(235,338),(235,375)])
box(a,235,515,400,70,'3. 步长生成\n'+r'$\Delta=\mathrm{softplus}(\mathrm{Linear}(\rho))$',PURPLE,12)
arrow(a,[(235,455),(235,480)])
box(a,575,525,240,330,'4. 二维卷积分支\n'+
    '还原空间排列\n深度卷积\nGroupNorm\n逐点卷积扩展\nGELU\n逐点卷积压缩\n展平并转置\n'+r'$U=\mathrm{MedNeXtConv}_{2D}(T)$',ORANGE,11)
arrow(a,[(455,315),(455,338),(575,338),(575,360)])
box(a,235,660,400,160,'4—5. 内层门控\n'+r'$A=-\exp(A_{\log})$'+'\n'+
    r'$G_U=\sum_{n=1}^N\exp(\Delta A_n)$'+'\n'+
    r'$G_T=\Delta$',ORANGE,13)
arrow(a,[(235,550),(235,580)])
box(a,355,810,500,80,'5. 内层门控融合\n'+r'$Y_0=G_U\odot U+G_T\odot T$',ORANGE,14)
arrow(a,[(235,740),(235,770)])
arrow(a,[(575,690),(575,770)])
arrow(a,[(155,280),(15,280),(15,810),(105,810)])
txt(a,35,767,r'$T$',13)
box(a,355,930,500,80,'5. 内容直连\n'+r'$Y_1=Y_0+D\odot T$',ORANGE,14)
arrow(a,[(355,850),(355,890)])
arrow(a,[(15,810),(15,930),(105,930)])
box(a,355,1050,500,80,'6. 外层门控\n'+r'$Y_2=Y_1\odot\mathrm{SiLU}(Z)$',PURPLE,14)
arrow(a,[(355,970),(355,1010)])
arrow(a,[(690,315),(760,315),(760,1050),(605,1050)])
box(a,355,1170,500,80,'7. 输出投影与残差\n'+r'$Y=X+\mathrm{Linear}(Y_2)$',GREEN,14)
arrow(a,[(355,1090),(355,1130)])
arrow(a,[(605,85),(785,85),(785,1170),(605,1170)])
box(a,355,1260,500,50,'转置并还原为特征图',GREEN,13)
arrow(a,[(355,1210),(355,1235)])
save(f,'09_第三版候选_去除BC分支')
print(OUT/'09_第三版候选_去除BC分支.svg')

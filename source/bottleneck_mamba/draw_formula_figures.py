"""Draw the code-checked Mamba-1 explanation with fixed orthogonal connectors."""
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


# One complete diagram. Keep the handwritten symbols; equations live in the boxes.
f,a=canvas(1170)
a.set_ylim(1100,-70)
box(a,355,-40,320,42,'特征图 → 序列 · 展平并转置',BLUE,12)
arrow(a,[(355,-19),(355,10)])
box(a,355,30,160,40,r'$X$',BLUE,16)
box(a,355,110,500,70,
    '1. 输入投影与拆分\n'+r'$(F,Z)=\mathrm{Split}(\mathrm{Linear}(\mathrm{LN}(X)))$',PURPLE,12)
arrow(a,[(355,50),(355,75)])
box(a,355,205,430,64,
    '2. 因果 DWConv1d（核4）+ SiLU\n'+r'$T=\mathrm{SiLU}(\mathrm{DWConv}(F))$',PURPLE,13)
arrow(a,[(355,145),(355,173)])
box(a,690,205,130,64,'门控\n'+r'$\mathrm{SiLU}(Z)$',BLUE,13)
arrow(a,[(605,110),(690,110),(690,173)])
box(a,355,310,500,90,
    '2. 线性变换与拆分\n'+r'$Q=\mathrm{Linear}(T)$'+'\n'+
    r'$(\rho,B,C)=\mathrm{Split}(Q;32,16,16)$',PURPLE,13)
arrow(a,[(355,237),(355,265)])
box(a,355,420,500,70,
    '3. 步长\n'+r'$\Delta=\mathrm{softplus}(\mathrm{Linear}(\rho))$',PURPLE,13)
arrow(a,[(355,355),(355,385)])
box(a,355,550,530,120,
    '4. 状态更新\n'+r'$A=-\exp(A_{\log})\qquad h_0=0$'+'\n'+
    r'$h_t=\exp(\Delta_t A)\odot h_{t-1}+\Delta_t B_tT_t$',ORANGE,13)
arrow(a,[(355,455),(355,490)])
# B and C leave the projection result; T has an independent D skip.
arrow(a,[(105,310),(75,310),(75,550),(90,550)])
txt(a,62,465,r'$B$',13)
arrow(a,[(75,310),(45,310),(45,685),(105,685)])
txt(a,33,640,r'$C$',13)
box(a,355,685,500,120,
    '5. 状态读出（N = 16）\n'+r'$Y_0=\sum_{n=1}^{N}C_nh_n$',ORANGE,15)
arrow(a,[(355,610),(355,625)])
box(a,355,790,500,66,
    '5. 内容直连\n'+r'$Y_1=Y_0+D\odot T$',ORANGE,14)
arrow(a,[(355,745),(355,757)])
arrow(a,[(140,205),(20,205),(20,790),(105,790)])
txt(a,51,748,r'$T$',13)
box(a,355,885,500,66,
    '6. 门控\n'+r'$Y_2=Y_1\odot\mathrm{SiLU}(Z)$',PURPLE,14)
arrow(a,[(355,823),(355,852)])
arrow(a,[(690,237),(690,885),(605,885)])
box(a,355,985,500,66,
    '7. 输出投影与残差\n'+r'$Y=X+\mathrm{Linear}(Y_2)$',GREEN,14)
arrow(a,[(355,918),(355,952)])
arrow(a,[(435,30),(780,30),(780,985),(605,985)])
txt(a,748,948,r'$X$',13)
box(a,355,1065,320,44,'序列 → 特征图 · 转置并还原',GREEN,12)
arrow(a,[(355,1018),(355,1043)])
save(f,'04_原始Mamba完整公式总图')
print(OUT/'04_原始Mamba完整公式总图.svg')

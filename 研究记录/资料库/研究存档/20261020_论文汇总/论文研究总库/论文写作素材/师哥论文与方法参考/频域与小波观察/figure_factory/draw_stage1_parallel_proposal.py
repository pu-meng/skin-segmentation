"""Design diagrams only: frozen original blocks plus three parallel residual kernels."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Circle

BASE=Path(__file__).resolve().parents[1];OUT=BASE/'figures'
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','svg.fonttype':'path','axes.unicode_minus':False})
INK='#29384b';LINE='#566577';BLUE='#5e7d96';PURPLE='#8554b0'
FROZEN='#eaf1f6';LEARN='#f1e7fc';FEATURE='#e1f3f3';GREEN='#e4f3e8'
checks=[]

def canvas(h,title,subtitle):
    fig,ax=plt.subplots(figsize=(11,h));fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
    ax.set(xlim=(0,11),ylim=(0,h));ax.axis('off')
    txt(ax,5.5,h-.45,title,21,True);txt(ax,5.5,h-.95,subtitle,13)
    return fig,ax

def txt(ax,x,y,s,size=14,bold=False,color=INK,**kw):
    return ax.text(x,y,s,ha='center',va='center',fontsize=size,fontweight='bold' if bold else 'normal',color=color,**kw)

def box(ax,x,y,w,h,title,detail,fill=FROZEN,edge=BLUE):
    p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.015,rounding_size=.12',facecolor=fill,edgecolor=edge,lw=1.6,zorder=3)
    ax.add_patch(p)
    a=txt(ax,x+w/2,y+h*.66,title,16,True,zorder=4)
    b=txt(ax,x+w/2,y+h*.27,detail,13,zorder=4)
    checks.append((ax,p,a,b))

def arrow(ax,points,color=LINE):
    if len(points)>2:
        x,y=zip(*points[:-1]);ax.plot(x,y,color=color,lw=1.6,zorder=2)
    ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=15,lw=1.6,
                                color=color,shrinkA=0,shrinkB=0,zorder=2))

def plus(ax,x,y):
    ax.add_patch(Circle((x,y),.23,facecolor='white',edgecolor=PURPLE,lw=1.8,zorder=4))
    txt(ax,x,y,'+',20,True,zorder=5)

def save(fig,ax,name):
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for owner,p,*labels in checks:
        if owner is not ax:continue
        bb=p.get_window_extent(renderer)
        for t in labels:
            b=t.get_window_extent(renderer)
            assert b.x0>bb.x0+3 and b.x1<bb.x1-3,t.get_text()
            assert b.y0>bb.y0+2 and b.y1<bb.y1-2,t.get_text()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT/(name+'.svg'))
    fig.savefig('/tmp/'+name+'.png',dpi=800/11)
    plt.close(fig)

# 1. Whole network: stage1 contains the modified blocks, all native skip connections stay.
fig,ax=canvas(14,'图1  新方案总图：只在编码第1层加入修正','设计图，尚未实现／训练；不再使用旧的单个无QK分支')
box(ax,1.0,11.35,3.7,1,'原始CT','[B,1,128,128,128]',FEATURE)
box(ax,1.0,9.85,3.7,1,'原stem：冻结','1×1×1 卷积，1 → 32')
arrow(ax,[(2.85,11.35),(2.85,10.85)])
arrow(ax,[(2.85,9.85),(2.85,9.25)])
ys=[8.7,6.95,5.2,3.45]
for i,y in enumerate(ys):
    if i==0:box(ax,1.0,y-.55,3.7,1.1,'第1层：原block＋修正','原block冻结；修正可学',LEARN,PURPLE)
    else:box(ax,1.0,y-.55,3.7,1.1,f'编码第{i+1}层：冻结',f'{32*2**i}×{128//2**i}³')
    box(ax,7.1,y-.55,3.1,1.1,f'解码第{i+1}层：冻结',f'{32*2**i}×{128//2**i}³')
    arrow(ax,[(4.7,y),(7.1,y)])
    txt(ax,5.9,y+.22,'原skip保留',12,color=BLUE)
    if i<3:
        arrow(ax,[(2.85,y-.55),(2.85,ys[i+1]+.55)])
        arrow(ax,[(8.65,ys[i+1]+.55),(8.65,y-.55)])
box(ax,3.65,1.35,3.7,1.05,'原瓶颈卷积＋MHA','512×8³；冻结')
arrow(ax,[(2.85,2.9),(2.85,1.875),(3.65,1.875)])
arrow(ax,[(7.35,1.875),(8.65,1.875),(8.65,2.9)])
box(ax,7.1,9.85,3.1,1,'原分割头：冻结','32 → 3')
arrow(ax,[(8.65,9.25),(8.65,9.85)])
box(ax,7.1,11.35,3.1,1,'分割输出','3类分数 → 预测标签',GREEN)
arrow(ax,[(8.65,10.85),(8.65,11.35)])
txt(ax,5.5,.81,'每级解码：上采样 → 与skip相加 → 原解码块。',13)
txt(ax,5.5,.39,'编码第1层内部展开见图2；每个新增模块内部展开见图3。',13)
save(fig,ax,'stage1_parallel_v2_overview')

# 2. All three wrappers: source block first, independent correction added to source output.
fig,ax=canvas(17.6,'图2  第1个stage：原block输出＋新增修正','全程保持 [B,32,128,128,128]；三个修正模块参数各自独立')
box(ax,4.4,15.4,3.8,.95,'来自原stem的特征','32通道；位置网格保持不变',FEATURE)
arrow(ax,[(6.3,15.4),(6.3,14.85)])
box(ax,.55,15.4,3.1,.95,'原CT＋spacing','只供灰度／距离权重使用',FEATURE)
# Guidance bus is distinct from the feature path, so CT cannot be confused with values.
ax.plot([.25,.25],[15.0,3.7],color=PURPLE,lw=1.4,zorder=1)
arrow(ax,[(.55,15.875),(.25,15.875),(.25,15.0)],PURPLE)
for i,top in enumerate((14.85,10.55,6.25),start=1):
    box(ax,4.4,top-1,3.8,1,f'原block F{i}：冻结','原卷积、归一化与残差均保留')
    py=top-2.45
    # Source block output splits into identity and a trainable branch.
    arrow(ax,[(6.3,top-1),(6.3,py+.23)])
    ax.plot([6.3],[top-1.4],'o',ms=3.5,color=LINE,zorder=4)
    box(ax,.65,py-.55,3.65,1.1,f'新增修正 R{i}：可训练','三路并联；输出同尺寸Δ',LEARN,PURPLE)
    arrow(ax,[(6.3,top-1.4),(2.475,top-1.4),(2.475,py+.55)],PURPLE)
    arrow(ax,[(.25,py),(.65,py)],PURPLE)
    plus(ax,6.3,py)
    arrow(ax,[(4.3,py),(6.07,py)],PURPLE)
    txt(ax,5.15,py+.25,f'Δ{i}',14,color=PURPLE)
    txt(ax,8.85,py+.12,f'Y{i}＋Δ{i}',16,True)
    if i<3:arrow(ax,[(6.3,py-.23),(6.3,top-4.3)])
    else:arrow(ax,[(6.3,py-.23),(6.3,1.25)])
box(ax,4.4,.3,3.8,.95,'原down0 → 编码第2层','由32×128³变为64×64³')
txt(ax,2.45,1.2,'γ初始为0：Δ初始为0',13,color=PURPLE)
save(fig,ax,'stage1_parallel_v2_insertions')

# 3. Branch-level diagram: independent matrices aggregate the same current feature Y.
fig,ax=canvas(14,'图3  每个修正模块：QK、距离、灰度三路并联','三路分别归一化、分别聚合同一份原block输出Y，再乘通道权重')
box(ax,3.7,11.65,3.6,.95,'原block输出 Y','[B,32,D,H,W]',FEATURE)
box(ax,7.85,11.65,2.65,.95,'原CT＋spacing','与Y使用同一网格',FEATURE)
centers=[1.95,5.5,9.05];left=[.4,3.95,7.5];w=3.1
# Shared Y feeds Q/K and all value-aggregation boxes; no learned V projection in this proposal.
arrow(ax,[(5.5,11.65),(5.5,11.1)])
ax.plot([.24,9.05],[11.1,11.1],color=BLUE,lw=1.6,zorder=2)
ax.plot([5.5],[11.1],'o',ms=3,color=BLUE,zorder=4)
arrow(ax,[(1.95,11.1),(1.95,10.55)])
box(ax,left[0],9.45,w,1.1,'Q、K从Y产生','QK^T / √d → softmax',LEARN,PURPLE)
box(ax,left[1],9.45,w,1.1,'毫米距离权重','距离越小，权重越大',LEARN,PURPLE)
box(ax,left[2],9.45,w,1.1,'CT灰度权重','灰度差越小，权重越大',LEARN,PURPLE)
# CT is used only by the two prior branches, not as the feature value being aggregated.
arrow(ax,[(9.175,11.65),(9.175,11.43),(10.75,11.43),(10.75,9.9),(10.6,9.9)],PURPLE)
arrow(ax,[(10.75,10.9),(6.7,10.9),(6.7,10.55)],PURPLE)
for i,x in enumerate(centers):
    # A_j and Y are explicit separate inputs to the aggregation box.
    box(ax,left[i],6.9,w,1.15,['QK加权原特征','距离加权原特征','灰度加权原特征'][i],['Aq × Y','Ad × Y','Ag × Y'][i],LEARN,PURPLE)
    arrow(ax,[(x,9.45),(x,8.05)],PURPLE)
    txt(ax,x+.28,8.63,['Aq','Ad','Ag'][i],13,color=PURPLE)
    # Local feature feed rail starts at shared-Y split, outside the weight boxes.
    gutter=left[i]-.16
    arrow(ax,[(x,11.1),(gutter,11.1),(gutter,7.48),(left[i],7.48)],BLUE)
    box(ax,left[i],4.8,w,1.05,['乘γq','乘γd','乘γg'][i],'32个通道权重，初始为0',LEARN,PURPLE)
    arrow(ax,[(x,6.9),(x,5.85)],PURPLE)
plus(ax,5.5,3.45)
arrow(ax,[(1.95,4.8),(1.95,3.45),(5.27,3.45)],PURPLE)
arrow(ax,[(5.5,4.8),(5.5,3.68)],PURPLE)
arrow(ax,[(9.05,4.8),(9.05,3.45),(5.73,3.45)],PURPLE)
box(ax,3.7,1.4,3.6,1.05,'输出修正量 Δ','返回图2的加号，与Y相加',LEARN,PURPLE)
arrow(ax,[(5.5,3.22),(5.5,2.45)],PURPLE)
txt(ax,5.5,.88,'每个Y位置对应一个32维特征；×表示空间权重矩阵乘特征。',12)
txt(ax,5.5,.42,'采用局部窗口，具体窗口大小与Q/K维度待实现时确定。',12)
save(fig,ax,'stage1_parallel_v2_branches')
print('Saved 3 design SVGs; all box text fits.')

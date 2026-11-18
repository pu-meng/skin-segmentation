"""Draw current implemented trainer, preserving frozen-backbone and skip semantics."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Circle
from matplotlib import font_manager

BASE=Path(__file__).resolve().parents[1]
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(FONT)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=FONT).get_name(),
                     'svg.fonttype':'path','axes.unicode_minus':False})
fig,ax=plt.subplots(figsize=(11,18))
fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
ax.set(xlim=(0,11),ylim=(0,18));ax.axis('off')
TEXT='#253247';LINE='#526172';FROZEN='#eaf0f6';BLUE='#5c7890';TRAIN='#f1e7ff';PURPLE='#8355ad';INPUT='#e1f3f4';OUTPUT='#e4f3e7'
labels=[]
def text(x,y,s,size=14,weight='normal',color=TEXT,**kw):
    return ax.text(x,y,s,ha='center',va='center',fontsize=size,fontweight=weight,color=color,**kw)
def box(x,y,w,h,title,detail,fill=FROZEN,edge=BLUE):
    patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015,rounding_size=.12',
                         lw=1.6,edgecolor=edge,facecolor=fill,zorder=3)
    ax.add_patch(patch)
    a=text(x+w/2,y+h*.66,title,16,'bold',zorder=4)
    b=text(x+w/2,y+h*.28,detail,13,zorder=4)
    labels.append((patch,a,b))
    return patch

def arrow(points,color=LINE,lw=1.7):
    if len(points)>2:
        xs,ys=zip(*points[:-1]);ax.plot(xs,ys,color=color,lw=lw,zorder=2,solid_joinstyle='round')
    ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=15,
                                lw=lw,color=color,shrinkA=0,shrinkB=0,zorder=2))

text(5.5,17.63,'早期距离＋灰度分支：当前 trainer 总图',22,'bold')
text(5.5,17.13,'只在 stem 后加一条分支；原编码器、瓶颈 MHA、解码器及 skip 保留',13)
box(4,15.95,3,1,'输入 CT','[B,1,128,128,128]',INPUT,BLUE)
box(4,14.25,3,.95,'原 stem（冻结）','1×1×1 卷积：1 → 32')
arrow([(5.5,15.95),(5.5,15.2)])
# Raw CT guidance is separate from the stem's feature/value input.
arrow([(4,16.45),(.28,16.45),(.28,12.6),(.75,12.6)],PURPLE)
text(.12,14.65,'CT 灰度（归一化）',12,color=PURPLE,rotation=90)
box(.75,14.7,2.9,.95,'预处理后 spacing','D/H/W：1/.768/.768 mm',INPUT,BLUE)
arrow([(1.5,14.7),(1.5,13.2)],PURPLE)
box(.75,12.0,3.95,1.2,'新增：无 Q/K 分支','距离＋灰度权重 × V → 修正量 δ',TRAIN,PURPLE)
text(2.72,11.58,'可训练：876 参数；局部窗口 4×4×4',13,color=PURPLE)
# Features are forked both to the branch and the identity path.
arrow([(5.5,14.25),(5.5,12.83)])
ax.plot([5.5],[13.8],'o',ms=4,color=LINE,zorder=4)
arrow([(5.5,13.8),(2.72,13.8),(2.72,13.2)],PURPLE)
text(3.85,14.0,'32 通道特征',12,color=PURPLE)
ax.add_patch(Circle((5.5,12.6),.23,facecolor='white',edgecolor=PURPLE,lw=1.8,zorder=4))
text(5.5,12.6,'+',20,'bold',zorder=5)
arrow([(4.7,12.6),(5.27,12.6)],PURPLE)
text(4.97,12.94,'δ',15,color=PURPLE)
text(6.55,12.68,'逐位置相加',13)
text(6.55,12.27,'32×128³',13)
arrow([(5.5,12.37),(5.5,11.22),(2.7,11.22),(2.7,10.7)])
# Frozen source model. Decoder boxes aggregate UpBlock -> skip sum -> decoder blocks.
ax.add_patch(FancyBboxPatch((.58,2.15),9.95,8.85,boxstyle='round,pad=.02,rounding_size=.18',
                           facecolor='#f8fafc',edgecolor='#b8c5d1',lw=1.4,zorder=0))
levels=[(10.2,32,128),(8.5,64,64),(6.8,128,32),(5.1,256,16)]
for i,(y,c,n) in enumerate(levels):
    box(1,y-.5,3.4,1,f'编码第{i+1}层 · E{i}',f'{c}×{n}³')
    box(7.15,y-.5,3.05,1,f'解码第{i+1}层 · D{i}',f'{c}×{n}³')
    arrow([(4.4,y),(7.15,y)])
    text(5.77,y+.2,'原 skip',12,color=BLUE)
    if i<3:
        nexty=levels[i+1][0]
        arrow([(2.7,y-.5),(2.7,nexty+.5)])
        text(3.23,(y+nexty)/2,'↓×2',12)
        arrow([(8.675,nexty+.5),(8.675,y-.5)])
box(3.8,2.65,3.65,1.25,'瓶颈卷积＋原 MHA','512×8³；全部冻结')
arrow([(2.7,4.6),(2.7,3.275),(3.8,3.275)])
arrow([(7.45,3.275),(8.675,3.275),(8.675,4.6)])
# The head and output remain frozen source behavior, drawn on the upward decoder path.
box(7.25,13.65,2.85,.95,'原分割头（冻结）','32 → 3')
arrow([(8.675,10.7),(8.675,13.65)])
box(7.25,15.2,2.85,1,'分割输出','3 类分数 → 预测标签',OUTPUT,BLUE)
arrow([(8.675,14.6),(8.675,15.2)])
text(5.5,1.72,'每级解码框内部：先上采样 → 与对应 skip 相加 → 解码块',14)
text(5.5,1.28,'层内尺寸简写为 C×D³；本图输入块为 128³，批量 B=2。',12)
text(5.5,.79,'紫色：新增且可学习     蓝灰色：源模型冻结     标签只用于训练损失',13)
text(5.5,.37,'均匀权重对照：只把紫色分支中的权重改为均匀，位置、V 和 MLP 相同。',12)
# Source text must fit its module. This catches accidental clipping after text changes.
fig.canvas.draw();renderer=fig.canvas.get_renderer()
for patch,*texts in labels:
    bounds=patch.get_window_extent(renderer)
    for t in texts:
        b=t.get_window_extent(renderer)
        assert b.x0>bounds.x0+4 and b.x1<bounds.x1-4,(t.get_text(),'horizontal overflow')
        assert b.y0>bounds.y0+2 and b.y1<bounds.y1-2,(t.get_text(),'vertical overflow')
out=BASE/'figures';out.mkdir(exist_ok=True)
fig.savefig(out/'early_bilateral_overview.svg')
fig.savefig('/tmp/early_bilateral_overview.png',dpi=90)
plt.close(fig)
print(out/'early_bilateral_overview.svg')

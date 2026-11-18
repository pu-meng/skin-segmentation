"""补充总结中的有限轮同步更新候选；未实现或训练的设计图。"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
ROOT=Path(__file__).resolve().parents[1]
FONT=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
font_manager.fontManager.addfont(str(FONT))
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(FONT)).get_name(),'svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(12,9))
fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
ax.set(xlim=(0,1000),ylim=(750,0));ax.axis('off')
INK='#293e55'
def txt(x,y,s,size=15,bold=False,color=INK):
 ax.text(x,y,s,ha='center',va='center',fontsize=size,fontweight='bold' if bold else 'normal',color=color)
def box(x,y,w,h,title,sub,color='#e7f1fc'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=7',facecolor=color,edgecolor=INK,lw=1.3))
 txt(x+w/2,y+h*.33,title,16,True)
 txt(x+w/2,y+h*.72,sub,13)
def arrow(pts,color=INK):
 if len(pts)>2:ax.plot([x for x,y in pts[:-1]],[y for x,y in pts[:-1]],color=color,lw=1.5)
 ax.add_patch(FancyArrowPatch(pts[-2],pts[-1],arrowstyle='-|>',mutation_scale=14,lw=1.5,color=color,shrinkA=0,shrinkB=0))
txt(500,35,'图5｜每轮先读旧状态，三个节点算完再一起替换',20,True)
txt(500,74,'人工候选：X是固定外来特征；A、B、I是特征图状态，不是生物电脉冲。',13)
box(300,120,400,82,'上一轮状态：A_t、B_t、I_t','初始A_0、B_0、I_0全为零')
for x,title,sub in [(40,'更新A','读取X与旧B_t'),(360,'更新B','读取X、旧A_t与旧I_t'),(680,'更新I','只读取旧A_t')]:
 box(x,290,280,84,title,sub,'#f6e8a6' if title!='更新I' else '#e6d5f7')
 arrow([(500,202),(500,246),(x+140,246),(x+140,290)])
 arrow([(x+140,374),(x+140,418),(500,418),(500,465)])
box(300,465,400,82,'三个新状态全部就绪，再一起替换','A_new、B_new、I_new成为下一轮状态')
arrow([(700,505),(985,505),(985,161),(700,161)],'#7e57a6')
txt(843,570,'未到T轮：继续更新',13,color='#7e57a6')
arrow([(500,547),(500,622)])
txt(365,587,'到达T轮',13)
box(300,622,400,82,'输出最终状态：A_T、B_T、I_T','怎样融合并接回U-Net，仍需另行固定','#cdefd8')
txt(500,735,'T是预设的有限轮数；本图没有加入“下游→A”的连接。候选尚未实现或验证。',13)
out=ROOT/'figures/大脑网络_图5_有限轮同步更新候选.svg'
fig.savefig(out,facecolor='white')
preview=Path('/tmp/brain_network_notes_preview/大脑网络_图5_有限轮同步更新候选.png')
preview.parent.mkdir(exist_ok=True,parents=True)
fig.savefig(preview,dpi=95,facecolor='white')
plt.close(fig)
print(out)

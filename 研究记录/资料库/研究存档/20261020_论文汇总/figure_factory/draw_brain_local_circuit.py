"""按用户截图重绘兴奋—抑制局部回路；示意连接，不是实测脑区模型。"""
from pathlib import Path
from math import hypot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch

ROOT=Path(__file__).resolve().parents[1]
FONT=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
font_manager.fontManager.addfont(str(FONT))
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(FONT)).get_name(),'svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(10,10.5))
fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
ax.set(xlim=(0,1000),ylim=(1050,0),aspect='equal'); ax.axis('off')
INK='#172a3d'; BLUE='#2d68c4'; GOLD='#bd8616'

def text(x,y,s,size=19,bold=False,color=INK):
    ax.text(x,y,s,ha='center',va='center',fontsize=size,fontweight='bold' if bold else 'normal',color=color,zorder=6)

def rect(x,y,w,h,s):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=12',facecolor='white',edgecolor=INK,lw=1.6))
    text(x+w/2,y+h/2,s)

def arrow(p,q,color=INK,rad=0):
    ax.add_patch(FancyArrowPatch(p,q,connectionstyle=f'arc3,rad={rad}',arrowstyle='-|>',mutation_scale=20,lw=2,color=color,shrinkA=0,shrinkB=0))

def boundary(p,q,r):
    d=hypot(q[0]-p[0],q[1]-p[1])
    return (p[0]+r*(q[0]-p[0])/d,p[1]+r*(q[1]-p[1])/d)

text(500,55,'图2(b)｜局部生物回路：兴奋与抑制怎样连接',21,True)
text(500,95,'按提供的截图布局整理；圆圈表示教学用神经元。',13)
A=(260,370); B=(740,370); I=(500,620); R=68
rect(410,170,180,80,'外来输入')
# 外来输入分别投向A、B，终点落在圆周。
for p,target in [((450,250),A),((550,250),B)]: arrow(p,boundary(target,p,R))
# A、B双向兴奋连接：分开两条弧，避免把往返画成一条线。
arrow((324,347),(676,347),BLUE,-.24)
arrow((676,393),(324,393),BLUE,-.24)
text(500,280,'A → B：兴奋',16,color=BLUE)
text(500,465,'B → A：兴奋',16,color=BLUE)
# A兴奋I。
arrow(boundary(A,I,R),boundary(I,A,R))
text(290,570,'A 激活 I',16)
# I抑制B，横杠终点表示抑制，并与箭头语义区分。
p=boundary(I,B,R); q=boundary(B,I,R)
ax.plot([p[0],q[0]],[p[1],q[1]],color=GOLD,lw=2.4)
dx,dy=q[0]-p[0],q[1]-p[1]; d=hypot(dx,dy)
nx,ny=-dy/d,dx/d
ax.plot([q[0]-12*nx,q[0]+12*nx],[q[1]-12*ny,q[1]+12*ny],color=GOLD,lw=3,zorder=5)
text(710,570,'I 抑制 B',16,color=GOLD)
# A、B也分别向下游输出，沿外侧绕开局部回路。
arrow(boundary(A,(130,650),R),(445,845),rad=.40)
arrow(boundary(B,(870,650),R),(555,845),rad=-.40)
rect(410,845,180,80,'下游')
for xy,title in [(A,'兴奋 A'),(B,'兴奋 B'),(I,'抑制 I')]:
    ax.add_patch(Circle(xy,R,facecolor='white',edgecolor=INK,lw=1.7,zorder=4))
    text(*xy,title,19)
text(500,969,'箭头：影响／传递方向；T形端点：抑制性作用',14)
text(500,1004,'示意连接，不是特定脑区的完整模型；线长不代表传递时间。',13)
out=ROOT/'figures/大脑网络_图2b_兴奋抑制局部回路.svg'
preview=Path('/tmp/brain_network_notes_preview/大脑网络_图2b_兴奋抑制局部回路.png')
preview.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(out,facecolor='white')
fig.savefig(preview,dpi=110,facecolor='white')
plt.close(fig)
print(out)

"""Code-faithful network diagrams for the seed42 100-epoch frozen-UNet probe."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Circle
from matplotlib.font_manager import FontProperties

OUT=Path(__file__).resolve().parents[3] / 'log/UNet_冻结模块频谱对比/figures'
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['svg.fonttype']='path'
FROZEN='#eaf1f8';TRAIN='#fff0d7';SAMPLE='#e8f5eb';INK='#243647'

def text(ax,x,y,s,size=12,**kw):ax.text(x,y,s,fontproperties=font,fontsize=size,ha='center',va='center',color=INK,**kw)
def box(ax,x,y,s,w=3,h=.85,fc=FROZEN,size=12):
 ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.05,rounding_size=0.12',ec='#6a7c89',fc=fc,lw=1.2,zorder=3));text(ax,x,y,s,size,zorder=4)
def arrow(ax,a,b,label=None,dashed=False):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,lw=1.3,color='#667985',linestyle='--' if dashed else '-',zorder=1))
 if label:text(ax,(a[0]+b[0])/2,(a[1]+b[1])/2+.16,label,10)
def save(fig,name):
 fig.savefig(OUT/(name+'.svg'),bbox_inches='tight',facecolor='white');fig.savefig(OUT/(name+'.png'),dpi=150,bbox_inches='tight',facecolor='white');plt.close(fig)

fig,ax=plt.subplots(figsize=(13,11.8));ax.set(xlim=(0,12),ylim=(0,12));ax.axis('off')
text(ax,6,11.65,'100轮频谱实验总图：冻结 UNet-2D + 瓶颈探针',17)
text(ax,6,11.12,'seed42；三组各自训练，图中 P 每次只选一种模块；不是三路融合',11)
box(ax,2,10.35,'输入图像\n3×256×256',w=2.9,h=.8,fc='white')
box(ax,10,10.35,'分割概率图\n1×256×256',w=2.9,h=.8,fc=SAMPLE)
left=[('编码1：32×256×256',9),('编码2：64×128×128',7.6),('编码3：128×64×64',6.2),('编码4：256×32×32',4.8),('编码5：512×16×16',3.4)]
right=[('解码4：32×256×256',9),('解码3：64×128×128',7.6),('解码2：128×64×64',6.2),('解码1：256×32×32',4.8)]
for label,y in left:box(ax,2,y,label,size=14)
for label,y in right:box(ax,10,y,label,size=14)
arrow(ax,(2,9.9),(2,9.48))
for i in range(4):arrow(ax,(2,left[i][1]-.46),(2,left[i+1][1]+.46),None)
text(ax,.57,5.5,'2×\n最大池化',10)
text(ax,11.55,9.7,'1×1 Conv\n+ Sigmoid',9)
for i in range(3):arrow(ax,(10,right[i+1][1]+.46),(10,right[i][1]-.46))
arrow(ax,(10,9.46),(10,9.9))
for _,y in right:arrow(ax,(3.58,y),(8.42,y),dashed=True)
text(ax,6,8.35,'虚线：原 UNet 跳跃连接',11)
text(ax,6,6.9,'编码器、解码器及 BatchNorm\n权重与统计全部固定',12)
text(ax,6,5.5,'编码块：两层 Conv-BN-ReLU\n解码：转置卷积放大 → 拼接 skip\n→ 两层 Conv-BN-ReLU',11)
box(ax,6,3.4,'瓶颈探针 P（下图展开）\nConv / Attention / Mamba\nY = X + 0.1×branch',w=3.3,h=1.15,fc=TRAIN)
arrow(ax,(3.58,3.4),(4.25,3.4));arrow(ax,(7.73,3.4),(10,3.4));arrow(ax,(10,3.4),(10,4.32))
text(ax,2,2.45,'X：同一瓶颈特征',11)
box(ax,6,1.55,'预测 + 真实分割标签 → BCE + Dice 损失\n梯度穿过冻结解码器，只更新 mixer 和 FFN',w=10.3,h=1.2,fc='white',size=13)
text(ax,6,.4,'蓝色：固定部分    橙色：含可训练参数    维度：通道×高×宽（省略 batch）',11)
save(fig,'seed42_100轮_网络总图')

fig,ax=plt.subplots(figsize=(13,13));ax.set(xlim=(0,12),ylim=(0,13.7));ax.axis('off')
text(ax,6,13.25,'瓶颈探针完整结构与频谱取样位置',17)
steps=[(12.15,'X：编码器第5块输出，512×16×16\n这是原瓶颈特征，还没有进入新增分支',FROZEN),
(10.85,'固定 1×1 Conv：512→128\n展开成 256 个 token，再做无仿射 LayerNorm',FROZEN),
(9.45,'T：固定降维＋LayerNorm的输出\n模块的输入，128×16×16；主频谱灰线测这里',SAMPLE),
(8.05,'M(T)：所选空间模块的直接输出，128×16×16\n主频谱彩线测这里；此时尚未加 T，也未加 X',TRAIN),
(6.65,'Z1 = T + M(T)',FROZEN),
(5.25,'F = FFN(LayerNorm(Z1))\n可训练 FFN：128→256→128',TRAIN),
(3.85,'Z2 = Z1 + F；随后减去 T\n送往上投影的是 Z2 − T = M(T) + F',FROZEN),
(2.45,'branch：M(T)＋FFN结果，经固定升维后的修正量\n512×16×16；没有加 X，不是分割预测',FROZEN),
(1.05,'Y：整个探针的输出，512×16×16\nY = X + 0.1×branch；加回 X 后送入解码器',FROZEN)]
for y,label,color in steps:box(ax,5.3,y,label,w=7.5,h=.9,fc=color,size=12)
for i in range(len(steps)-1):arrow(ax,(5.3,steps[i][0]-.49),(5.3,steps[i+1][0]+.49))
text(ax,10.5,9.45,'主频谱输入\n来自降维＋归一化\n不是原图或 X',12)
text(ax,10.5,8.05,'主频谱输出\n来自空间模块\n不是 branch 或 Y',12)
arrow(ax,(9.12,9.45),(9.65,9.45));arrow(ax,(9.12,8.05),(9.65,8.05))
text(ax,10.5,6.5,'FFT在取样后计算\n比较 T 与 M(T)\n\n不含后续 FFN\n不含外层残差',11)
text(ax,10.5,2.45,'另存修正量\n主曲线不取这里',10)
text(ax,10.5,1.05,'另存加回后的特征\n主曲线不取这里',10)
text(ax,5.5,.15,'仅模块 M 和 FFN 可训练；投影固定，LayerNorm无可训练参数。',11)
save(fig,'seed42_100轮_分支与频谱取样')


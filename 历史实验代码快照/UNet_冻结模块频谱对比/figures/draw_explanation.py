from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Circle
from matplotlib.font_manager import FontProperties

OUT=Path(__file__).resolve().parent
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

fig,ax=plt.subplots(figsize=(11,10));ax.set(xlim=(0,12),ylim=(0,12));ax.axis('off')
text(ax,6,11.65,'本轮试验总图：冻结 UNet-2D，仅训练新增分支',17)
text(ax,6,11.12,'三组独立参数：Conv / Self-Attention / Mamba；下图一次表示其中一组',11)
box(ax,2,10.35,'输入图像\n3×256×256',w=2.9,h=.8,fc='white')
box(ax,10,10.35,'分割概率图\n1×256×256',w=2.9,h=.8,fc=SAMPLE)
left=[('E0：32×256×256',9),('E1：64×128×128',7.6),('E2：128×64×64',6.2),('E3：256×32×32',4.8),('E4：512×16×16',3.4)]
right=[('D0：32×256×256',9),('D1：64×128×128',7.6),('D2：128×64×64',6.2),('D3：256×32×32',4.8)]
for label,y in left:box(ax,2,y,label)
for label,y in right:box(ax,10,y,label)
arrow(ax,(2,9.9),(2,9.48))
for i in range(4):arrow(ax,(2,left[i][1]-.46),(2,left[i+1][1]+.46),None)
text(ax,.57,5.5,'2×\n下采样',10)
text(ax,11.55,9.7,'1×1 Conv\n+ Sigmoid',9)
for i in range(3):arrow(ax,(10,right[i+1][1]+.46),(10,right[i][1]-.46))
arrow(ax,(10,9.46),(10,9.9))
for _,y in right:arrow(ax,(3.58,y),(8.42,y),dashed=True)
text(ax,6,8.35,'虚线：原 UNet 跳跃连接',11)
text(ax,6,6.9,'编码器、解码器及 BatchNorm\n权重与统计全部固定',12)
text(ax,6,5.5,'解码步骤：上采样 → 拼接 skip\n→ 两层 Conv-BN-ReLU',11)
box(ax,6,3.4,'新增分支 P（下图展开）\n输出 Y = X + 0.1×branch',w=3.3,h=1,fc=TRAIN)
arrow(ax,(3.58,3.4),(4.25,3.4));arrow(ax,(7.73,3.4),(10,3.4));arrow(ax,(10,3.4),(10,4.32))
text(ax,2,2.45,'X：同一瓶颈特征',11)
box(ax,6,1.55,'预测 + 真实分割标签 → BCE + Dice 损失\n梯度穿过冻结解码器，只更新 mixer 和 FFN',w=10.3,h=1.2,fc='white',size=13)
text(ax,6,.4,'蓝色：固定部分    橙色：含可训练参数    维度：通道×高×宽（省略 batch）',11)
save(fig,'网络总图')

fig,ax=plt.subplots(figsize=(11,11));ax.set(xlim=(0,12),ylim=(0,13.7));ax.axis('off')
text(ax,6,13.25,'新增分支内部与频谱取样位置',17)
steps=[(12.15,'X：UNet 瓶颈特征 512×16×16',FROZEN),
(10.85,'固定 1×1 Conv：512→128\n展开成 256 个 token，再做无仿射 LayerNorm',FROZEN),
(9.45,'T：256×128（还原为 128×16×16）\n三组完全相同的 mixer 输入',SAMPLE),
(8.05,'M(T)：可训练 mixer\nConv / 四头 Self-Attention / 原生 Mamba',TRAIN),
(6.65,'Z1 = T + M(T)',FROZEN),
(5.25,'F = FFN(LayerNorm(Z1))\n可训练 FFN：128→256→128',TRAIN),
(3.85,'Z2 = Z1 + F\nZ2 − T = M(T) + F',FROZEN),
(2.45,'固定 1×1 Conv：128→512\n得到 branch：512×16×16',FROZEN),
(1.05,'Y = X + 0.1×branch → 冻结解码器',FROZEN)]
for y,label,color in steps:box(ax,5.3,y,label,w=7.5,h=.9,fc=color,size=12)
for i in range(len(steps)-1):arrow(ax,(5.3,steps[i][0]-.49),(5.3,steps[i+1][0]+.49))
text(ax,10.5,9.45,'取样 A\n输入 T',12)
text(ax,10.5,8.05,'取样 B\n直接输出 M(T)',12)
arrow(ax,(9.12,9.45),(9.65,9.45));arrow(ax,(9.12,8.05),(9.65,8.05))
text(ax,10.5,6.5,'主频谱表\n比较 A 与 B\n\n不含 FFN\n不含外层残差',11)
text(ax,10.5,2.45,'另存 branch',10)
text(ax,10.5,1.05,'另存 Y',10)
text(ax,5.5,.15,'主表分析的是内部特征，不是最终分割概率图，也不是分割标签。',11)
save(fig,'分支与取样位置')

fig,axes=plt.subplots(1,2,figsize=(11,5),gridspec_kw={'width_ratios':[1,1.2]})
f=np.fft.fftshift(np.fft.fftfreq(16));fy,fx=np.meshgrid(f,f,indexing='ij');r=np.sqrt(fx**2+fy**2)
region=np.where(r<=.125,0,np.where(r<.25,1,2));region[r==0]=3
from matplotlib.colors import ListedColormap
axes[0].imshow(region,origin='lower',extent=[-.53125,.46875,-.53125,.46875],cmap=ListedColormap(['#79b7e3','#f7d788','#e99891','#222222']),vmin=0,vmax=3,interpolation='nearest')
for rad in [.125,.25]:axes[0].add_patch(Circle((0,0),rad,fill=False,ec='white',lw=1.5))
axes[0].set_xlabel('fx (cycles/feature-pixel)');axes[0].set_ylabel('fy (cycles/feature-pixel)');axes[0].set_title('16×16 特征的二维频率格点',fontproperties=font)
axes[1].axis('off')
for y,msg in [(0.88,'r = √(fx² + fy²)：离零频点的径向距离'),(.70,'黑点：r = 0，通道均值／直流，排除'),(.53,'蓝色：0 < r ≤ 0.125，低频'),(.36,'黄色：0.125 < r < 0.25，中频'),(.19,'红色：r ≥ 0.25，高频（包括方形频谱角落）'),(.01,'这些阈值是本试验固定设置，不是网络学出的。')]:
 axes[1].text(0,y,msg,fontproperties=font,fontsize=12,va='center')
fig.tight_layout();save(fig,'频段划分')

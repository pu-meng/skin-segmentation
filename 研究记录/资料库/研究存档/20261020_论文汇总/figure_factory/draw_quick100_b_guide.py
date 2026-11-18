"""Quick100 B: three source-checked SVGs for personal Markdown reading. No torch/GPU."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import xml.etree.ElementTree as ET
BASE=Path('/home/PuMengYu/论文汇总')
OUT=BASE/'figures/quick100_b'
PREVIEW=Path('/tmp/quick100_b_figure_preview')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','svg.fonttype':'none','font.size':12})
C={'feature':('#eaf4ff','#2563eb'),'block':('#fef9c3','#ca8a04'),'moe':('#f3e8ff','#9333ea'),'expert':('#fae8ff','#c026d3'),'output':('#ecfdf5','#047857'),'note':('#fff1f2','#be123c')}
def canvas(w,h,title):
 fig,ax=plt.subplots(figsize=(w,h));ax.set_xlim(0,w);ax.set_ylim(h,0);ax.axis('off')
 ax.text(w/2,.45,title,ha='center',va='center',fontsize=23,weight='bold')
 return fig,ax
def box(ax,x,y,w,h,title,sub='',kind='feature',fs=18,dashed=False):
 face,edge=C[kind]
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.025,rounding_size=.12',facecolor=face,edgecolor=edge,lw=1.5,linestyle='--' if dashed else '-'))
 ax.text(x+w/2,y+h*(.38 if sub else .5),title,ha='center',va='center',fontsize=fs,weight='bold')
 if sub:ax.text(x+w/2,y+h*.75,sub,ha='center',va='center',fontsize=fs-2,color='#334155')
 return (x,y,w,h)
def arr(ax,a,b,dashed=False,color='#334155'):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,lw=1.4,color=color,shrinkA=0,shrinkB=0,linestyle='--' if dashed else '-',zorder=4))
def path(ax,pts,**kw):
 for a,b in zip(pts[:-2],pts[1:-1]):ax.plot([a[0],b[0]],[a[1],b[1]],color=kw.get('color','#64748b'),lw=1.4,linestyle='--' if kw.get('dashed') else '-')
 arr(ax,pts[-2],pts[-1],**kw)
def plus(ax,x,y,r=.18):
 ax.add_patch(Circle((x,y),r,fc='white',ec='#334155',lw=1.5,zorder=5));ax.text(x,y,'+',ha='center',va='center',fontsize=19,zorder=6)
def save(fig,name):
 OUT.mkdir(parents=True,exist_ok=True);PREVIEW.mkdir(exist_ok=True)
 fig.savefig(OUT/(name+'.svg'),bbox_inches='tight',facecolor='white')
 fig.savefig(PREVIEW/(name+'.png'),dpi=100,bbox_inches='tight',facecolor='white')
 ET.parse(OUT/(name+'.svg'));plt.close(fig)
def overview():
 fig,ax=canvas(12,16,'图1　B组整网：MoE只在瓶颈内')
 stages=[(1.1,'CT patch','[B,1,128,128,128]','feature'),(2.7,'Stem：输入通道投影','1 → 32 通道，空间尺寸不变','block'),(4.3,'Encoder 0–3 ＋ 四次DownBlock','保留 E0–E3；最终 [B,512,8,8,8]','block'),(6.2,'MedNeXt瓶颈：8个Block','[B,512,8,8,8]','block'),(7.9,'两个 MHA＋MoE Transformer块','展开见图2；专家选择见图3','moe'),(9.8,'Decoder 3–0：四级恢复分辨率','每级 UpBlock → 加对应skip → Decoder','block'),(11.7,'输出头：1×1×1卷积','32 → 3 类分数；[B,3,128,128,128]','output'),(13.4,'分割结果','softmax／argmax：背景、非肿瘤肝脏、肿瘤','output')]
 for y,t,s,k in stages:box(ax,3.1,y,7.8,1.05,t,s,k)
 for a,b in zip(stages,stages[1:]):arr(ax,(7,a[0]+1.05),(7,b[0]))
 path(ax,[(3.1,4.82),(1.25,4.82),(1.25,10.33),(3.1,10.33)])
 ax.text(1.4,7.3,'原始skip\nE0–E3',ha='left',va='center',fontsize=16,color='#475569')
 ax.text(6,15.2,'B继续训练改变权重，结构不变。\n图示为128³输入；形状中的B为批大小。',ha='center',fontsize=15,color='#475569')
 save(fig,'01_B组完整网络')
def blocks():
 fig,ax=canvas(11,26.5,'图2　瓶颈：两次MHA、两次MoE，各有残差')
 box(ax,3,1.1,6,1,'瓶颈三维特征','[B,512,8,8,8]')
 box(ax,3,2.7,6,1,'展平空间位置＋转置','[B,N,C]；N=512，C=512')
 arr(ax,(6,2.1),(6,2.7))
 start=4.3
 for block in [0,1]:
  y=start+block*9
  ax.add_patch(FancyBboxPatch((.4,y-.3),10.1,8.6,boxstyle='round,pad=.02,rounding_size=.12',fc='#f8fafc',ec='#cbd5e1',lw=1,zorder=-1))
  ax.text(.7,y+.05,f'Block {block+1}（代码索引 {block}）',fontsize=16,weight='bold')
  box(ax,3,y+.5,6,.85,f'输入 X{block}','[B,N,512]')
  box(ax,3,y+1.9,6,.85,'LayerNorm → MHA','8 heads；Q/K/V与输出投影','block')
  plus(ax,6,y+3.4)
  box(ax,3,y+4.1,6,.85,'LayerNorm → MoE-FFN','1共享专家＋4候选中的Top-2','moe')
  plus(ax,6,y+5.6)
  box(ax,3,y+6.3,6,.85,f'输出 X{block+1}','[B,N,512]','output')
  for a,b in [(y+1.35,y+1.9),(y+2.75,y+3.22),(y+3.58,y+4.1),(y+4.95,y+5.42),(y+5.78,y+6.3)]:arr(ax,(6,a),(6,b))
  path(ax,[(3,y+.92),(1.3,y+.92),(1.3,y+3.4),(5.82,y+3.4)])
  path(ax,[(6,y+3.7),(1.8,y+3.7),(1.8,y+5.6),(5.82,y+5.6)])
  ax.text(9.3,y+3.4,'相加',va='center',fontsize=15)
  ax.text(9.3,y+5.6,'相加',va='center',fontsize=15)
 arr(ax,(6,3.7),(6,start+.5))
 arr(ax,(6,start+7.15),(6,start+9+.5))
 y=start+9
 box(ax,3,22.1,6,1,'最终LayerNorm＋还原三维形状','[B,512,8,8,8]','output')
 arr(ax,(6,y+7.15),(6,22.1))
 box(ax,2,24,8,1.2,'诊断先检查最后一个MoE块','只临时关闭某候选专家输出；参数不更新','note',dashed=True)
 save(fig,'02_B组瓶颈两块')
def moe():
 fig,ax=canvas(16,18,'图3　MoE内部：同一份特征，三条并行处理路径')
 box(ax,5,1.1,6,1,'展平 LayerNorm 2 输出','[B×N,512]')
 for x in [2.5,8,13.1]:path(ax,[(8,2.1),(8,2.8),(x,2.8),(x,3.5)])
 box(ax,.5,3.5,4,1.15,'共享专家','512 → 1024 → 512','expert')
 box(ax,5.5,3.5,5,1.15,'4个候选专家：0、1、2、3','各自独立参数；全部计算','expert')
 box(ax,11.2,3.5,3.8,1.15,'Router：评分器','Linear 512 → 4','moe')
 box(ax,5.5,5.7,5,1,'四份候选输出','[B×N,4,512]','expert')
 arr(ax,(8,4.65),(8,5.7))
 box(ax,10.9,5.7,4.4,1,'原分数＋均衡偏置 → Top-2','只产生两个选中索引','moe',fs=16)
 arr(ax,(13.1,4.65),(13.1,5.7))
 box(ax,5.5,8.2,5,1,'按索引取得两个专家输出','[B×N,2,512]','expert')
 arr(ax,(8,6.7),(8,8.2))
 path(ax,[(10.9,6.2),(10.65,6.2),(10.65,7.6),(9.5,7.6),(9.5,8.2)])
 box(ax,11.1,8.2,4.2,1,'取原分数 → softmax','两个权重；不加均衡偏置','moe',fs=16)
 arr(ax,(13.1,6.7),(13.1,8.2))
 path(ax,[(15,4.1),(15.65,4.1),(15.65,7.6),(14.6,7.6),(14.6,8.2)])
 ax.text(15.55,6.9,'原分数',rotation=90,fontsize=14,ha='center',va='center',color='#475569')
 box(ax,5.5,10.7,8.5,1,'两个选中输出乘权重后求和','得到 routed output：[B×N,512]','moe')
 arr(ax,(8,9.2),(8,10.7))
 path(ax,[(13.2,9.2),(13.2,10.15),(12.6,10.15),(12.6,10.7)])
 plus(ax,8,13)
 path(ax,[(2.5,4.65),(2.5,5.05),(.18,5.05),(.18,13),(7.82,13)])
 arr(ax,(8,11.7),(8,12.82))
 ax.text(8.4,13,'共享输出＋路由输出',va='center',fontsize=16)
 box(ax,5.5,14,6,1,'还原 [B,N,512]','返回图2，进入MoE外层残差相加','output')
 arr(ax,(8,13.18),(8,14))
 box(ax,.4,7.1,4.2,2,'拟进行的专家移除诊断','置零某专家输出；索引不变\n不补选、不重新归一化','note',fs=16,dashed=True)
 path(ax,[(4.6,8.1),(4.95,8.1),(4.95,7.15),(8,7.15)],dashed=True,color='#be123c')
 ax.text(8,16,'每个专家：Linear → GELU → Dropout → Linear → Dropout',ha='center',fontsize=16)
 ax.text(8,16.65,'512 → 1024 → 512；Dropout=0。\n共享专家与四个候选专家的参数相互独立。',ha='center',fontsize=15,color='#475569')
 ax.text(8,17.6,'红色虚线表示拟做诊断，不是B组原有网络层。',ha='center',fontsize=15,color='#be123c')
 save(fig,'03_B组专家与评分')
if __name__=='__main__':
 overview();blocks();moe();print(OUT);print(PREVIEW)

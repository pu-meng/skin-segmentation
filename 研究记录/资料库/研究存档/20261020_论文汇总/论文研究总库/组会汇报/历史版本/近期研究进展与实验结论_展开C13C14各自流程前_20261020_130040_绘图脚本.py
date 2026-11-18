"""Code-checked skip2 correction: three projections/experts, original skip, decoder upsample.
Source: MultiStageSkipCorrection and SkipOnlyCorrection; input patch128, base32.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Circle
OUT=Path('/home/PuMengYu/论文汇总/论文研究总库/组会汇报/figures/方法_skip2接入与三路分支细节.svg')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','svg.fonttype':'none','axes.unicode_minus':False})
fig=plt.figure(figsize=(800/72,790/72));ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,800);ax.set_ylim(790,0);ax.axis('off');checks=[]
def text(x,y,label,size=17,bold=False):
 return ax.text(x,y,label,ha='center',va='center',fontsize=size,fontweight='bold' if bold else 'normal',color='#26394a')
def box(x,y,w,h,title,subtitle,color='#e6f0fb',size=16.5):
 p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=8',linewidth=1.5,edgecolor='#52687c',facecolor=color);ax.add_patch(p)
 ts=[text(x+w/2,y+h*.33,title,size,True),text(x+w/2,y+h*.73,subtitle,15.5)]
 checks.append((p,ts));return p
def arrow(a,b):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,linewidth=1.6,color='#52687c',shrinkA=1,shrinkB=1))
def plus(x,y):
 ax.add_patch(Circle((x,y),14,edgecolor='#52687c',facecolor='white',linewidth=1.5));text(x,y,'+',21)
text(400,22,'skip2：只接入1个融合模块，内部有3条分支',20,True)
xs=[110,340,570];centers=[205,435,665]
for x,c,title,shape in zip(xs,centers,['第2级编码特征 E1','第3级编码特征 E2','第4级编码特征 E3'],['64通道，64³','128通道，32³','256通道，16³']):
 box(x,85,190,60,title,shape)
 arrow((c,145),(c,185))
for x,c,title,shape in zip(xs,centers,['池化＋通道投影','保持大小＋通道投影','插值＋通道投影'],['64→128，32³','128→128，32³','256→128，32³']):
 box(x,185,190,70,title,shape,'#e8f4f1',15.5)
 arrow((c,255),(c,295))
for i,(x,c) in enumerate(zip(xs,centers),1):
 box(x,295,190,70,f'专家分支{i}','128→32→GELU→128','#eeebf8',16.5)
 arrow((c,365),(c,405))
box(110,405,650,65,'三份专家结果各乘比例，再相加','C13：全局共用；C14：逐位置计算（见图1）','#fff0db',18)
arrow((435,470),(435,536));text(570,500,'三路补充量：128通道、32³',16)
# Original E2 bypass goes above and outside the three processing branches.
ax.plot([435,435,780,780],[85,60,60,550],color='#52687c',linewidth=1.6)
arrow((780,550),(449,550));text(655,533,'原E2直接保留',16)
plus(435,550)
arrow((435,564),(435,626));text(565,592,'修正后的skip2',17)
box(70,607,220,66,'解码器上采样特征','128通道，32³','#e6f0fb',16.5)
arrow((290,640),(421,640));plus(435,640)
arrow((435,654),(435,700))
box(310,700,250,65,'对应解码块','继续恢复空间与预测类别','#e7f4eb',18)
fig.canvas.draw();ren=fig.canvas.get_renderer()
for p,ts in checks:
 bb=p.get_window_extent(ren)
 for t in ts:
  tb=t.get_window_extent(ren)
  assert tb.x0>=bb.x0+4 and tb.x1<=bb.x1-4,t.get_text()
  assert tb.y0>=bb.y0+2 and tb.y1<=bb.y1-2,t.get_text()
fig.savefig(OUT,format='svg');fig.savefig('/tmp/skip2_insertion_detail_preview.png',dpi=59.4)
print(OUT)

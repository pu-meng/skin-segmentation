"""Readable diagrams for tensor axes, shared/spatial branch weights and full-network insertion.
Code: SkipOnlyCorrection.forward and MultiStageSkipCorrection.routing_weights.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Circle
DEST=Path('/home/PuMengYu/论文汇总/论文研究总库/组会汇报/figures')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','svg.fonttype':'none','axes.unicode_minus':False})
class Diagram:
 def __init__(self,w,h):
  self.fig=plt.figure(figsize=(w/72,h/72));self.ax=self.fig.add_axes([0,0,1,1]);self.ax.set_xlim(0,w);self.ax.set_ylim(h,0);self.ax.axis('off');self.checks=[]
 def text(self,x,y,s,size=18,bold=False,color='#26394a'):
  return self.ax.text(x,y,s,ha='center',va='center',fontsize=size,fontweight='bold' if bold else 'normal',color=color)
 def box(self,x,y,w,h,title,subtitle=None,color='#e7f0fa',size=18,subsize=17):
  p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=8',linewidth=1.5,edgecolor='#52687c',facecolor=color);self.ax.add_patch(p)
  ts=[self.text(x+w/2,y+h*(.34 if subtitle else .5),title,size,True)]
  if subtitle:ts.append(self.text(x+w/2,y+h*.74,subtitle,subsize))
  self.checks.append((p,ts));return p
 def arrow(self,a,b,color='#52687c',dash=False):
  self.ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,linewidth=1.6,color=color,linestyle='--' if dash else '-',shrinkA=1,shrinkB=1))
 def path(self,pts,color='#52687c',dash=False):
  self.ax.plot(*zip(*pts[:-1]),color=color,linewidth=1.6,linestyle='--' if dash else '-');self.arrow(pts[-2],pts[-1],color,dash)
 def plus(self,x,y):
  self.ax.add_patch(Circle((x,y),12,facecolor='white',edgecolor='#52687c',linewidth=1.5));self.text(x,y,'+',20)
 def save(self,name,preview):
  self.fig.canvas.draw();r=self.fig.canvas.get_renderer()
  for p,ts in self.checks:
   b=p.get_window_extent(r)
   for t in ts:
    a=t.get_window_extent(r)
    assert a.x0>=b.x0+4 and a.x1<=b.x1-4,t.get_text()
    assert a.y0>=b.y0+2 and a.y1<=b.y1-2,t.get_text()
  self.fig.savefig(DEST/name,format='svg');width=800 if '比例计算流程' in name else (720 if '整网' in name else 660);self.fig.savefig(preview,dpi=width/self.fig.get_figwidth())
  plt.close(self.fig);print(DEST/name)

# Comparison: literal tensor dimensions, exact sharing range and numerical examples.
d=Diagram(760,835)
d.text(380,24,'“共用”和“逐位置”：究竟在哪些维度上变？',21,True)
d.box(20,60,720,80,'专家特征：[B, 128, 32, 32, 32]',
      'B份CT；128个通道；深D、高H、宽W各32个位置',subsize=18)
d.text(195,177,'C13：全局共用',20,True);d.text(565,177,'C14：逐位置计算',20,True)
d.box(20,204,350,67,'训练学出3个分数 → softmax','得到一组三个比例；不读取当前CT','#eeebf8',17.5,16.5)
d.box(390,204,350,67,'三路编码＋解码 → 512通道','卷积512→32→GELU→卷积32→3','#e7f0fa',17.5,16.5)
d.arrow((195,271),(195,310));d.arrow((565,271),(565,310))
d.box(20,310,350,76,'同一组三个数在各处重复使用','广播至[B, 3, 32, 32, 32]','#fff0db',17.5,17)
d.box(390,310,350,76,'softmax(dim=1)','[B, 3, 32, 32, 32]，共32³组','#fff0db',17.5,17)
d.text(380,416,'p＝(深5, 高8, 宽9)，q＝(深5, 高8, 宽10)',17)
for y,label,left,right in [
 (455,'患者A，位置p','(0.2, 0.5, 0.3)','(0.2, 0.5, 0.3)'),
 (494,'患者A，位置q','(0.2, 0.5, 0.3)','(0.6, 0.3, 0.1)'),
 (533,'患者B，位置p','(0.2, 0.5, 0.3)','(0.1, 0.2, 0.7)')]:
 d.text(380,y,label,16)
 d.text(175,y,left,18,True);d.text(585,y,right,18,True)
d.box(20,575,720,76,'每组的3个比例分别对应专家分支1、2、3',
       '同一位置的128个通道共用该组三个比例','#e7f4eb',19,18)
d.text(380,684,'示意：同一通道的三份专家值为10、20、30',18)
d.text(380,716,'C13在p、q都得21；C14在p得21，在q得15',18,True)
d.text(380,750,'C13：比例参数3个；C14：比例网络参数16,515个',17)
d.text(380,782,'C14每份CT输出32³×3＝98,304个比例值；这些是输出，不是新增参数。',15.5)
d.text(380,815,'比例和专家值均仅作示意；3对应专家分支，不是类别数或空间轴。',16)
d.save('方法_C13_C14共享范围与位置维度.svg','/tmp/skip_sharing_axes_preview.png')

# Group each decoder stage: upsampling, skip addition and decoder block.
d=Diagram(800,900)
d.text(400,24,'整网总图：上采样与解码合为一个阶段',22,True)
d.box(20,65,200,55,'输入CT','1通道，128³',size=18,subsize=17)
d.box(485,65,295,45,'分割头 → 背景／肝脏／肿瘤',size=17)
levels=[180,350,520,690]
for i,(y,c,n) in enumerate(zip(levels,[32,64,128,256],[128,64,32,16])):
 d.box(20,y-32,200,64,f'编码第{i+1}级 E{i}',f'{c}通道，{n}³',size=18,subsize=17)
 d.box(485,y-42,295,84,f'解码第{i+1}级 D{i}', '上采样 → ＋skip → 解码块',size=19,subsize=17)
 if i<3:
  d.arrow((120,y+32),(120,levels[i+1]-32))
  d.text(157,y+84,'下采样',15)
  d.arrow((632,levels[i+1]-42),(632,y+42))
 if i!=2:d.arrow((220,y),(485,y))
 else:
  d.arrow((220,y),(280,y))
  d.box(280,y-36,170,72,'新增skip2模块','原E2＋三路补充','#fff0db',17,16)
  d.arrow((450,y),(485,y))
d.arrow((120,120),(120,148));d.arrow((632,138),(632,110))
# Direct cross-scale arrows, no stair-step routing.
d.arrow((220,350),(330,484),color='#aa6b23')
d.arrow((220,690),(330,556),color='#aa6b23')
d.arrow((485,550),(450,550),color='#805b9b',dash=True)
d.text(438,587,'仅C14读取U2',15,color='#805b9b')
d.path([(120,722),(120,805),(260,805)])
d.text(158,767,'下采样',15)
d.box(260,770,280,70,'瓶颈：MedNeXt＋MHA','512通道，8³；普通前馈层','#eeebf8',18,17)
d.path([(540,805),(632,805),(632,732)])
d.text(400,863,'橙线：E1、E3补充；紫虚线：本级上采样后的U2送给C14',16)
d.save('方法_skip融合整网位置总图.svg','/tmp/skip_network_overview_preview.png')

# Contrast the complete ratio-generation paths; no duplicated magnified inset.
d=Diagram(900,865)
d.text(450,25,'C13与C14：比例怎样产生？',23,True)
d.text(205,73,'C13：全局比例',22,True)
d.text(665,73,'C14：逐位置比例',22,True)
d.box(25,110,360,68,'3个可学习分数','形状：[3]','#eeebf8',20,19)
d.arrow((205,178),(205,210))
d.box(25,210,360,60,'softmax(dim=0)',color='#fff0db',size=20)
d.arrow((205,270),(205,310))
d.box(25,310,360,68,'得到一组三个比例','形状：[3]','#fff0db',20,19)
d.arrow((205,378),(205,410))
d.box(25,410,360,68,'扩展至全部CT和空间位置','[B,3,D,H,W]',size=19,subsize=19)
d.arrow((205,478),(205,610))
d.box(25,610,360,68,'所有CT、所有位置','共用一组三个比例','#fff0db',20,19)
# C14 changes the channel axis while preserving B and the spatial axes.
d.box(455,110,420,68,'F1、F2、F3、U2四份特征','每份：[B,128,D,H,W]',size=19,subsize=19)
d.arrow((665,178),(665,210))
d.box(455,210,420,68,'沿C轴拼接：cat(dim=1)','[B,512,D,H,W]',size=19,subsize=19)
d.arrow((665,278),(665,310))
d.box(455,310,420,68,'1×1×1卷积512→32 → GELU','[B,32,D,H,W]',size=18.5,subsize=19)
d.arrow((665,378),(665,410))
d.box(455,410,420,68,'1×1×1卷积32→3','[B,3,D,H,W]',size=20,subsize=19)
d.arrow((665,478),(665,510))
d.box(455,510,420,60,'softmax(dim=1)',color='#fff0db',size=20)
d.arrow((665,570),(665,610))
d.box(455,610,420,68,'得到比例：[B,3,D,H,W]','不同患者、不同位置的比例可以不同','#fff0db',19,17.5)
# Both outputs are broadcast to the expert channels when multiplying.
d.arrow((205,678),(205,730));d.arrow((665,678),(665,730))
d.box(25,730,850,82,'两者相同：三路专家结果分别乘对应比例，再相加',
      '每路比例[B,1,D,H,W] × 专家特征[B,128,D,H,W]',color='#e7f4eb',size=20,subsize=18)
d.text(450,841,'B＝CT数量；C＝通道数；D、H、W＝空间尺寸；本实验D＝H＝W＝32',17)
d.save('方法_C13_C14比例计算流程.svg','/tmp/c13_c14_ratio_comparison_preview.png')

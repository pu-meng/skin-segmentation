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
  self.fig.savefig(DEST/name,format='svg');width=720 if '整网' in name else 660;self.fig.savefig(preview,dpi=width/self.fig.get_figwidth())
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
d.box(390,310,350,76,'softmax：每位置的3路分数','[B, 3, 32, 32, 32]，共32³组','#fff0db',17.5,17)
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

# Full U-shaped network. Main decoder flow is bottom-up; every stage shows Up -> + -> block.
d=Diagram(800,1010)
d.text(400,22,'整网总图：新增模块只接在skip2',22,True)
d.box(20,65,200,55,'输入CT','1通道，128³',size=18,subsize=17)
d.box(510,50,200,40,'背景／肝脏／肿瘤',size=17)
levels=[190,380,570,760]
for i,(y,c,n) in enumerate(zip(levels,[32,64,128,256],[128,64,32,16])):
 title=f'编码第{i+1}级 E{i}'
 if i==0:title='stem＋编码第1级 E0'
 d.box(20,y-30,200,60,title,f'{c}通道，{n}³',size=17,subsize=17)
 if i<3:
  d.arrow((120,y+30),(120,y+70))
  d.box(65,y+70,110,40,'下采样',size=16)
  d.arrow((120,y+110),(120,levels[i+1]-30))
 d.box(535,y+40,150,40,f'上采样{i}',size=17)
 d.plus(610,y)
 d.arrow((610,y+40),(610,y+12))
 d.arrow((610,y-12),(610,y-32))
 d.box(510,y-78,200,46,f'解码第{i+1}级 D{i}',size=17)
 if i<3:d.arrow((610,levels[i+1]-78),(610,y+80))
 if i!=2:d.arrow((220,y),(598,y))
 else:
  d.arrow((220,y),(280,y))
  d.box(280,y-35,170,70,'新增skip2模块','原E2＋三路补充','#fff0db',17,16)
  d.arrow((450,y),(598,y))
d.arrow((120,120),(120,160));d.arrow((610,112),(610,90))
# Cross-scale inputs entering only the correction, leaving encoder descent intact.
d.path([(220,380),(250,380),(250,515),(330,515),(330,535)],color='#aa6b23')
d.path([(220,760),(250,760),(250,625),(390,625),(390,605)],color='#aa6b23')
# The decoder context is an input only to C14's ratio network, not to the three experts.
d.path([(535,630),(475,630),(475,592),(450,592)],color='#805b9b',dash=True)
d.text(495,656,'仅C14读取',16,color='#805b9b')
d.arrow((120,790),(120,815));d.box(65,815,110,40,'下采样',size=16)
d.path([(120,855),(120,910),(260,910)])
d.box(260,872.5,280,75,'瓶颈：MedNeXt＋MHA','512通道，8³；普通前馈层','#eeebf8',18,17)
d.path([(540,910),(610,910),(610,840)])
d.text(391,983,'橙线：E1、E3补充来源；紫色虚线：C14的解码上下文',16)
d.save('方法_skip融合整网位置总图.svg','/tmp/skip_network_overview_preview.png')

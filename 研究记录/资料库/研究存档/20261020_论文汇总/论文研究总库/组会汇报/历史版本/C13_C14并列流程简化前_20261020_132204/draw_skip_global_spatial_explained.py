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

# Tensor path and the channel vector at one fixed spatial coordinate.
d=Diagram(800,1200)
d.text(400,25,'C14：沿通道算比例，空间坐标保持不动',22,True)
d.text(400,61,'B＝CT数量；C＝通道数；D、H、W＝深、高、宽',17)
d.box(40,90,720,82,'四份输入：F1、F2、F3、U2，各为[B,128,D,H,W]',
      'F1～F3是对齐投影后的编码特征；U2是解码上采样特征',size=18,subsize=17)
d.arrow((400,172),(400,205))
d.box(40,205,720,76,'cat(dim=1)：仅拼接C轴 → X：[B,512,D,H,W]',
      '128＋128＋128＋128＝512；B、D、H、W均不变','#fff0db',18,17)
d.arrow((400,281),(400,315))
d.box(40,315,720,76,'1×1×1卷积512→32，再做GELU',
      '[B,512,D,H,W] → [B,32,D,H,W]',size=19,subsize=18)
d.arrow((400,391),(400,425))
d.box(40,425,720,76,'1×1×1卷积32→3：得到分支分数S',
      '[B,32,D,H,W] → [B,3,D,H,W]',size=19,subsize=18)
d.arrow((400,501),(400,535))
d.box(40,535,720,86,'softmax(dim=1)：每个位置只比较3条分支',
      'A：[B,3,D,H,W]；A[b,:,d,h,w]的3个数之和为1','#fff0db',18,17)
d.text(400,663,'把上面过程放大：固定第b份CT的一个位置p＝(d,h,w)',18,True)
d.box(40,695,720,174,'',color='#eeebf8')
d.text(400,722,'取X[b,:,d,h,w]：一个512维通道向量',19,True)
d.text(400,761,'32个中间值：各自对512个输入加权求和＋偏置，再做GELU',17)
d.text(400,800,'3个分数：各自对32个中间值加权求和＋偏置',17)
d.text(400,840,'对这3个分数做softmax → 此位置的3个分支比例',18,True)
d.text(400,900,'所有位置共用同一套卷积参数；并行计算，无需逐体素写循环',16)
d.text(400,944,'1×1×1只读取该坐标的通道值；这一步不混合相邻空间坐标',17)
d.box(40,983,720,90,'第j路：A[:,j:j+1] × 专家输出Zj',
      '[B,1,D,H,W] × [B,128,D,H,W] → [B,128,D,H,W]',color='#e7f4eb',size=19,subsize=17)
d.text(400,1096,'广播：一个位置的1个分支比例，乘该路在此位置的全部128通道',16)
d.text(400,1140,'三路相加得补充量Δ → 加回原E2 → 与U2相加 → 解码块',18,True)
d.text(400,1175,'本实验D＝H＝W＝32；3是专家分支数，与分割类别数无关。',16)
d.save('方法_C14五维张量逐位置操作.svg','/tmp/c14_tensor_operations_preview.png')

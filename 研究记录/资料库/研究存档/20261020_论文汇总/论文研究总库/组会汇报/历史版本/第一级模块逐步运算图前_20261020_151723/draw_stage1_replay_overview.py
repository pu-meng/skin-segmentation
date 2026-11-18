"""Stage1 Replay v2: checked against GuidedStage1.forward and Stage1ParallelBlock3D.
Generate vectors only; never start or alter experiments.
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
  self.fig.savefig(DEST/name,format='svg');width=800;self.fig.savefig(preview,dpi=width/self.fig.get_figwidth())
  plt.close(self.fig);print(DEST/name)

# Overview: feature dimensions correspond to one 128-cubed training patch.
d=Diagram(900,1040)
d.text(450,25,'第一级并行分支Replay v2：整网总图',23,True)
d.text(450,62,'蓝色：源模型参数固定；橙色：新增模块参数训练',18)
d.box(20,95,330,65,'输入CT','[B,1,128,128,128]',size=20,subsize=18)
d.box(515,95,365,65,'分割头（固定）','背景／肝脏／肿瘤',size=20,subsize=18)
d.arrow((185,160),(185,195))
d.box(20,195,330,55,'原stem卷积1→32（固定）',size=19)
d.arrow((185,250),(185,285))
d.box(20,285,330,100,'编码第1级E0：3次接入',
      '原块固定；新增模块①②③训练','#fff0db',20,17)
d.text(185,404,'[B,32,128,128,128]',18)
levels=[335,505,675,845]
for i,(y,c,n) in enumerate(zip(levels,[32,64,128,256],[128,64,32,16])):
 if i:
  d.box(20,y-35,330,70,f'编码第{i+1}级E{i}（固定）',f'[B,{c},{n},{n},{n}]',size=20,subsize=18)
  start=420 if i==1 else levels[i-1]+35
  d.arrow((185,start),(185,y-35))
  if i>1:d.text(240,(start+y-35)/2,'下采样',16)
  else:d.text(240,447,'下采样',16)
 d.box(515,y-40,365,80,f'解码第{i+1}级D{i}（固定）',
       '上采样 → ＋skip → 解码块',size=20,subsize=18)
 d.arrow((350,y),(515,y))
 if i<3:d.arrow((697,levels[i+1]-40),(697,y+40))
d.arrow((697,295),(697,160))
d.path([(185,880),(185,945),(310,945)])
d.box(310,908,300,74,'瓶颈：MedNeXt＋MHA','512通道，8³；无MoE','#e7f0fa',19,18)
d.path([(610,945),(697,945),(697,885)])
d.text(450,1014,'①接在原块1后 → ②接在原块2后 → ③接在原块3后；三处参数独立',17)
d.save('方法_第一级Replay_v2_整网总图.svg','/tmp/stage1_replay_network.png')

# Internals of any one of the three independently trained inserted modules.
d=Diagram(900,1020)
d.text(450,25,'新增模块①／②／③：内部都是三路并行计算',23,True)
d.box(20,70,860,80,'输入：当前特征[B,32,128,128,128]＋对齐CT灰度',
      'CT灰度[B,1,128,128,128]；仅作引导，不是分割标注',size=20,subsize=18)
d.arrow((450,150),(450,190))
d.box(20,190,860,82,'划分4×4×4窗口；每个窗口64个位置',
      '每份CT有32³＝32,768个窗口；每窗特征Y为[64,32]',size=20,subsize=18)
d.arrow((450,272),(450,310))
d.box(20,310,860,72,'每个窗口读取：特征Y、CT灰度g、物理坐标r',
      '三路都在本窗口内加权汇总同一份Y，无V投影',size=20,subsize=18)
# One bus fans the same window inputs out to all three computations.
d.ax.plot([155,745],[430,430],color='#52687c',linewidth=1.6)
d.arrow((450,382),(450,430))
for x in [155,450,745]:d.arrow((x,430),(x,470))
d.box(20,470,270,115,'A：特征相似',
      'LN → Q/K：32→8','#fff0db',21,18)
d.text(155,613,'softmax(QK^T/√8) × Y',17)
d.box(315,470,270,115,'B：空间距离',
      '距离越近，权重越大','#fff0db',21,17)
d.text(450,613,'softmax(−距离²/2σ²) × Y',16.5)
d.box(610,470,270,115,'C：CT灰度相似',
      '灰度越近，权重越大','#fff0db',21,17)
d.text(745,613,'softmax(−灰度差²/2σ²) × Y',16)
# Join after the three result labels, with explicit arrows into the aggregation.
for x in [155,450,745]:d.arrow((x,641),(x,680))
d.box(20,680,860,86,'三路结果分别乘各自的32通道门控，再相加',
      '每路门控是32个可学习数；三路结果都为[64,32]','#fff0db',20,18)
d.arrow((450,766),(450,805))
d.box(20,805,860,86,'按窗口还原补充量Δ，再加回本模块的输入特征',
      '输出＝原特征＋Δ，仍为[B,32,128,128,128]','#e7f4eb',20,18)
d.text(450,933,'门控零初始化：起始Δ＝0；三处各674个参数，合计2,022个',18)
d.text(450,976,'每模块每份CT：32,768窗 × 三路；同样的计算连续执行3个模块',17.5)
d.save('方法_第一级Replay_v2_模块内部.svg','/tmp/stage1_replay_module.png')

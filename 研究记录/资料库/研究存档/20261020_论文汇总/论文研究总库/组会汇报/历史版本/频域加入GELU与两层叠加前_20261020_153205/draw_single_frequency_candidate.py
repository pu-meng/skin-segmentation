"""PROPOSED single-insertion FFT residual adapter, for review only.
Not the existing all-block compressed frequency_adapter implementation.
No experiment, queue or training implementation is changed by this script.
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
  self.fig.savefig(DEST/name,format='svg');width=900;self.fig.savefig(preview,dpi=width/self.fig.get_figwidth())
  plt.close(self.fig);print(DEST/name)

d=Diagram(900,1160)
d.text(450,25,'候选方案：第1个原块后接一个频域残差模块',22,True)
d.text(450,63,'待审图，未启动训练；蓝色参数固定，橙色参数训练',18)
d.box(25,100,325,65,'输入CT','[B,1,128,128,128]',size=20,subsize=18)
d.arrow((187,165),(187,200))
d.box(25,200,325,55,'原stem：1→32通道（固定）',size=19)
d.arrow((187,255),(187,290))
d.box(25,290,325,55,'E0的第1个原块（固定）',size=20)
d.arrow((187,345),(187,380))
d.box(25,380,325,75,'新增频域残差模块','x → x＋频域补充(x)','#fff0db',21,18)
d.arrow((187,455),(187,490))
d.box(25,490,325,70,'E0的第2、3个原块（固定）','输出仍为[B,32,128,128,128]',size=19,subsize=17)
d.text(187,584,'上面三原块均属于编码第1级E0',16)
d.box(515,100,360,94,'源模型：MedNeXt＋MHA','加载原始源权重；不含瓶颈MoE',size=20,subsize=17)
d.box(515,360,360,65,'分割头（固定）','背景／肝脏／肿瘤',size=20,subsize=18)
levels=[525,675,825,975]
for i,(y,c,n) in enumerate(zip(levels,[32,64,128,256],[128,64,32,16])):
 if i:
  d.box(25,y-35,325,70,f'编码第{i+1}级E{i}（固定）',f'[B,{c},{n},{n},{n}]',size=20,subsize=18)
  start=600 if i==1 else levels[i-1]+35
  d.arrow((187,start),(187,y-35));d.text(245,(start+y-35)/2,'下采样',15)
 d.box(515,y-40,360,80,f'解码第{i+1}级D{i}（固定）','上采样 → ＋skip → 解码块',size=20,subsize=18)
 d.arrow((350,y),(515,y))
 if i<3:d.arrow((695,levels[i+1]-40),(695,y+40))
d.arrow((695,485),(695,425))
d.path([(187,1010),(187,1090),(310,1090)])
d.box(310,1055,300,70,'瓶颈：MedNeXt＋MHA','512通道、8³；参数固定',size=19,subsize=18)
d.path([(610,1090),(695,1090),(695,1015)])
d.text(450,1140,'只接一处；直接相加，不拼接；本图按“第一个原块后”这一位置暂定。',17)
d.save('候选_单处频域残差_整网总图.svg','/tmp/frequency_candidate_network.png')

# Literal direct spectral multiplication: zero-initialized correction, no channel projections.
d=Diagram(900,1190)
d.text(450,25,'候选模块：x＋频域分支(x)，全程不拼接',23,True)
d.text(450,62,'暂定32通道 × 8频带＝256个可学习参数',19)
d.box(75,95,775,76,'输入特征x：[B,32,128,128,128]',
      '来自E0第1个原块；原特征沿左侧直接保留',size=21,subsize=18)
d.arrow((450,171),(450,215))
d.box(75,215,775,80,'三维实数FFT：仅变换D、H、W',
      '复数频谱F：[B,32,128,128,65]',size=21,subsize=19)
d.arrow((450,295),(450,518))
d.text(280,397,'同一通道内',18)
d.text(280,432,'不同频带可乘不同系数',18)
d.box(550,350,300,85,'θ：[32,8]（训练）','每通道8个频带参数','#fff0db',20,18)
d.arrow((700,435),(700,490))
d.box(550,490,300,80,'tanh(θ) → 频带插值','G：[1,32,128,128,65]','#fff0db',19,16.5)
d.arrow((550,530),(462,530))
d.ax.add_patch(Circle((450,530),12,facecolor='white',edgecolor='#52687c',linewidth=1.5));d.text(450,530,'×',21)
d.box(100,590,750,70,'频谱逐元素相乘：F补充＝F×G',
      'F补充[b,c,k]＝F[b,c,k] × G[0,c,k]',size=20,subsize=18)
d.arrow((450,542),(450,590))
d.arrow((450,660),(450,700))
d.box(100,700,750,80,'逆FFT：把频域补充变回空间特征',
      'Δx：[B,32,128,128,128]，输出为实数',size=21,subsize=19)
d.arrow((450,780),(450,838))
d.path([(75,133),(35,133),(35,850),(438,850)])
d.text(235,813,'原输入特征x直接保留',18,True)
d.plus(450,850)
d.arrow((450,862),(450,900))
d.box(100,900,750,80,'输出y＝x＋Δx',
      '对应通道、对应空间位置相加；形状不变','#e7f4eb',23,20)
d.box(100,1020,750,75,'乘法示意：某频率F＝2＋3i，系数G＝0.2',
      '频域补充＝0.4＋0.6i；逆FFT后，才加回原空间特征',color='#fff0db',size=19,subsize=18)
d.text(450,1134,'θ起始为0 → G＝0 → Δx＝0 → 初始y＝x；图中数字仅作示意。',18)
d.text(450,1170,'G是实数径向频率响应；B维广播共用；没有按患者新增参数。',17)
d.save('候选_单处频域残差_模块内部.svg','/tmp/frequency_candidate_module.png')

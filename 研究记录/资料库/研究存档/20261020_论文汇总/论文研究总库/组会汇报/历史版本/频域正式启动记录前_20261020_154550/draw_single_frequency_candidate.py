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
d.box(25,380,325,75,'两层频域残差模块','x → h1 → h2；每层含GELU','#fff0db',21,18)
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

# Two independent nonlinear frequency residual layers, inserted at a single site.
d=Diagram(900,1450)
d.text(450,26,'候选模块：两个频域残差层串联，每层含GELU',23,True)
d.text(450,65,'每层32×8＝256个参数；θ1、θ2独立，合计512个；不拼接',18)
d.box(100,100,520,65,'输入h0＝x','[B,32,128,128,128]',size=22,subsize=19)
for index,base in [(1,200),(2,840)]:
 previous=f'h{index-1}'
 d.text(765,base+15,f'第{index}层',22,True)
 d.arrow((360,165 if index==1 else 740),(360,base))
 d.box(100,base,520,65,f'三维FFT：F{index}＝FFT({previous})',
       '[B,32,128,128,65]，复数频谱',size=21,subsize=18)
 d.arrow((360,base+65),(360,base+100))
 d.box(100,base+100,520,65,f'频谱逐元素乘：F{index} × G{index}',
       '同通道、同频率相乘；B维广播',size=21,subsize=18)
 d.box(670,base+80,200,105,f'θ{index}：[32,8]',
       'tanh → 频带插值','#fff0db',21,17)
 d.arrow((670,base+132.5),(620,base+132.5))
 d.arrow((360,base+165),(360,base+200))
 d.box(100,base+200,520,65,'逆FFT：还原空间特征',
       '[B,32,128,128,128]，实数',size=22,subsize=18)
 d.arrow((360,base+265),(360,base+300))
 d.box(100,base+300,520,65,'GELU：作用于逆FFT后的实数特征',
       f'得到本层非线性补充量δ{index}',color='#fff0db',size=20,subsize=19)
 add_y=base+430
 d.arrow((360,base+365),(360,add_y-12))
 input_y=132.5 if index==1 else 707.5
 d.path([(100,input_y),(45,input_y),(45,add_y),(348,add_y)])
 d.text(195,add_y-27,f'保留本层输入{previous}',18,True)
 d.plus(360,add_y)
 d.arrow((360,add_y+12),(360,base+475))
 d.box(100,base+475,520,65,f'h{index}＝{previous}＋δ{index}',
       '[B,32,128,128,128]，形状不变','#e7f4eb',23,subsize=18)
 if index==1:d.text(765,base+507,'送入第2层',19,True)
d.text(450,1410,'最终输出y＝h2；两层θ均从0开始，GELU(0)＝0，因此初始y＝x。',18)
d.save('候选_单处频域残差_模块内部.svg','/tmp/frequency_candidate_module.png')

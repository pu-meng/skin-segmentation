"""Generate readable paper diagrams from reading_cards.json.

Run this file directly. SVG deliverables remain under figures_reading/;
actual SVG previews and a layout audit are written under /tmp.
"""
from pathlib import Path
import json,re,subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures_reading'
SPEC=json.loads((Path(__file__).parent/'reading_cards.json').read_text())
PREVIEW=Path('/tmp/reading_database_review')/f'scheme{SPEC["scheme"]}'
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
BOLD='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
INK='#263342'
COLORS=[('#eaf2fc','#3e6798'),('#f1ecfb','#7955a6'),('#fff2df','#a56b26'),('#eaf5ee','#467c5b')]
plt.rcParams['svg.fonttype']='path'
R=RendererAgg(800,800,72)
AUDIT=[]

def wrap(text,size,width,bold=False):
 f=FontProperties(fname=BOLD if bold else FONT,size=size)
 lines=[]
 for paragraph in text.split('\n'):
  line=''
  for token in re.findall(r'[A-Za-z0-9_.%/+-]+|.',paragraph):
   trial=line+token
   if line and R.get_text_width_height_descent(trial,f,False)[0]>width*72:
    lines.append(line.rstrip());line=token.lstrip()
   else:line=trial
  lines.append(line.rstrip())
 return '\n'.join(lines)

def text(ax,x,y,s,size=15.5,bold=False,**kw):
 return ax.text(x,y,s,fontproperties=FontProperties(fname=BOLD if bold else FONT),
                fontsize=size,color=INK,linespacing=1.3,**kw)

def box_height(head,desc,width):
 h=wrap(head,17,width-.36,True);d=wrap(desc,15.5,width-.36)
 return max(.86,.30*len(h.split('\n'))+.28*len(d.split('\n'))+.28)

def box(ax,cx,top,width,height,head,desc,palette):
 patch=FancyBboxPatch((cx-width/2,top-height),width,height,
        boxstyle='round,pad=0.012,rounding_size=0.09',
        facecolor=palette[0],edgecolor=palette[1],linewidth=1.2)
 ax.add_patch(patch)
 head=wrap(head,17,width-.36,True);desc=wrap(desc,15.5,width-.36)
 y=top-.13
 text(ax,cx,y,head,17,True,ha='center',va='top')
 y-=.30*len(head.split('\n'))+.06
 text(ax,cx,y,desc,15.5,ha='center',va='top')

def arrow(ax,start,end):
 ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=14,
                             linewidth=1.3,color='#586371',shrinkA=0,shrinkB=0))

def render(stem,title,subtitle,nodes,footer,arrows=True):
 title=wrap(title,18,7.4,True);subtitle=wrap(subtitle,14.5,7.4)
 footer=wrap(footer,14.5,7.25)
 header=.22+.33*len(title.split('\n'))+.14+.27*len(subtitle.split('\n'))+.22
 hs=[]
 for node in nodes:
  if isinstance(node,dict):hs.append(max(box_height(h,d,3.42) for h,d in node['branches']))
  else:hs.append(box_height(node[0],node[1],7.25))
 gap=.34 if arrows else .17
 height=header+sum(hs)+gap*(len(nodes)-1)+.24+.26*len(footer.split('\n'))+.22
 fig=plt.figure(figsize=(8,height),facecolor='white');ax=fig.add_axes([0,0,1,1])
 ax.set(xlim=(0,8),ylim=(0,height));ax.axis('off')
 y=height-.22
 text(ax,.3,y,title,18,True,va='top');y-=.33*len(title.split('\n'))+.14
 text(ax,.3,y,subtitle,14.5,va='top')
 top=height-header;prev=None
 for i,(node,h) in enumerate(zip(nodes,hs)):
  branches=isinstance(node,dict)
  xs=[2.08,5.92] if branches else [4]
  if prev is not None and arrows:
   px,pbottom=prev;mid=(pbottom+top)/2
   if len(px)==1 and len(xs)==1:
    arrow(ax,(4,pbottom-.025),(4,top+.025))
   else:
    # One shared horizontal bus, with visible vertical arrow segments.
    for x in px:ax.plot([x,x],[pbottom-.025,mid],color='#586371',lw=1.3)
    ax.plot([min(px+xs),max(px+xs)],[mid,mid],color='#586371',lw=1.3)
    for x in xs:arrow(ax,(x,mid),(x,top+.025))
  if branches:
   for k,((head,desc),x) in enumerate(zip(node['branches'],xs)):
    box(ax,x,top,3.42,h,head,desc,COLORS[k])
  else:box(ax,4,top,7.25,h,node[0],node[1],COLORS[i%4])
  prev=(xs,top-h);top-=h+gap
 text(ax,.375,top+gap-.24,footer,14.5,va='top')
 # Check actual rendered text extents. Wrapping is measured before drawing.
 fig.canvas.draw();renderer=fig.canvas.get_renderer();overflow=[]
 canvas=fig.bbox
 for obj in ax.texts:
  bb=obj.get_window_extent(renderer)
  if bb.x0<0 or bb.x1>canvas.x1 or bb.y0<0 or bb.y1>canvas.y1:overflow.append(obj.get_text())
 if overflow:raise ValueError((stem,'text outside canvas',overflow))
 OUT.mkdir(exist_ok=True);PREVIEW.mkdir(parents=True,exist_ok=True)
 svg=OUT/(stem+'.svg');fig.savefig(svg);plt.close(fig)
 subprocess.run(['rsvg-convert','-w','800','-o',str(PREVIEW/(stem+'.png')),str(svg)],check=True)
 AUDIT.append({'figure':svg.name,'text_overflow':overflow,'width_px':800,'height_in':round(height,2)})

def method_nodes(p):
 n=p['n'];s=SPEC['scheme']
 if s==5 and n==1:
  return [('同一输入图像','两条网络分别读取原始图像'),
   {'branches':[('原网络：冻结','保留已有参数，产生基础特征'),('侧网络：可训练','学习目标任务的补充特征')]},
   ('按学习比例加权相加','α×基础特征＋(1−α)×侧特征'),
   ('目标任务预测头','读取融合特征；标签损失训练侧网络、比例和头')]
 if s==5 and n==2:
  return [('同一输入，两项任务','任务A与B分别提取对应层特征'),
   {'branches':[('任务A特征','记为A'),('任务B特征','记为B')]},
   ('可学习2×2线性混合','A新＝a×A＋b×B；B新＝c×A＋d×B'),
   {'branches':[('A新特征','送入任务A的下一层'),('B新特征','送入任务B的下一层')]},
   ('两任务标签共同监督','两网络与四个混合系数一起学习')]
 return p['steps']

for p in SPEC['papers']:
 n=p['n'];short=p['name']
 render(f'Method_{n:03d}',f'{n}. {short}｜方法',p['question'],method_nodes(p),
        '测试：'+p['test'],True)
 render(f'Evidence_{n:03d}',f'{n}. {short}｜证据',
        '每张卡片是一项对照；先看固定了什么，再看结果。',p['evidence'],
        '数字为作者报告，口径见图下注释；没有重新运行原实验。',False)
for key,title,steps,bound in SPEC['concepts']:
 render('Explain_'+key,title,'看清这里固定了什么、改变了什么。',steps,bound,False)
(PREVIEW/'layout_audit.json').write_text(json.dumps(AUDIT,ensure_ascii=False,indent=2))
print('Scheme',SPEC['scheme'],': rendered',len(AUDIT),'SVGs at 800 px; canvas text bounds passed.')

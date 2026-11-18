"""E2.1-only figures, using the shared FigureStyle and its common config."""
from pathlib import Path
import sys
sys.path.insert(0,'/home/PuMengYu/.codex/skills/research-figure-style/scripts')
from figure_style import FigureStyle
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'研究记录/模块设计/figures/三模块'

def make(title,height):
 s=FigureStyle();f,a=s.canvas(height);s.text(a,400,30,title,role='heading');return s,f,a

def box(s,a,x,y,text,c='input'):return s.box(a,x,y,text,s.cfg['colors'][c])
def edge(s,a,n,m):s.arrow(a,[(n['x'],n['bottom']),(m['x'],m['top'])])
def control(s,a,points):
 old_lines=len(a.lines);old_patches=len(a.patches);s.arrow(a,points)
 for art in list(a.lines)[old_lines:]+list(a.patches)[old_patches:]:art.set_linestyle('--')
def save(s,f,name):
 s.compact_vertical(f);print(name,s.save(f,OUT/name)['passed'])

s,f,a=make('E2.1｜接入位置与完整外围',1240)
s.text(a,400,75,'原 Align3 → 新模块 → 原 D3；原四层 Decoder 保留',role='annotation')
items=[]
for y,text,color in [
 (145,'最深对齐特征 X\n[B,512,8,8]','input'),
 (285,'前置深度可分离卷积 → U\nDW3 → PW1：512→256\nBN → GELU；[B,256,8,8]','projection'),
 (435,'简化 Mixer(U)\n低频三支扫描＋高频三支卷积','state'),
 (580,r'$V=U+M(U)$','output'),
 (725,'FFN(V) · 保留4倍扩张\n1×1：256→1024→256','state'),
 (870,r'$W=V+\mathrm{FFN}(V)$','output'),
 (1015,'后置深度可分离卷积\nDW3 → PW1：256→512\nBN → GELU','projection'),
 (1155,'输出交给原 D3\nDoubleConv 512→512 保留','output')]:items.append(box(s,a,400,y,text,color))
for n,m in zip(items,items[1:]):edge(s,a,n,m)
s.arrow(a,[(items[1]['right'],285),(730,285),(730,580),(items[3]['right'],580)])
s.text(a,755,460,'U',role='dimension')
s.arrow(a,[(items[3]['left'],580),(70,580),(70,870),(items[5]['left'],870)])
s.text(a,45,735,'V',role='dimension')
s.text(a,400,1220,'两次内部残差保持；没有跨越前后卷积的额外 X 直连。',role='annotation')
save(s,f,'E2_1_01_完整外围与接入位置')

s,f,a=make('E2.1｜内部256通道，六个分支均带残差',1640)
s.text(a,400,77,'实线：内容；虚线：跨频控制；圆圈＋：分支残差',role='annotation')
start=box(s,a,400,160,'U：[B,256,8,8] → 互补FFT圆形分频\n低频半径≤0.25；高频为补集','projection')
fused=[];gated=[]
from matplotlib.patches import Circle
for x,name,ops,alpha in [
 (200,'低频 L0',['A · 四方向 Mamba\n86通道','B · 四方向 Mamba\n85通道','C · 四方向 Mamba\n85通道'],r'$\alpha_L$'),
 (600,'高频 H0',['A · RepDW3\n86通道','B · RepDW7\n85通道','C · 膨胀 DW3\ndilation=2；85通道'],r'$\alpha_H$')]:
 inp=box(s,a,x,285,name+'\n1×1：256→256 → Split\n86 / 85 / 85','projection')
 s.arrow(a,[(400,start['bottom']),(400,220),(x,220),(x,inp['top'])])
 bus=x-160;collector=x+155;results=[]
 for y,text in zip([430,625,820],ops):
  n=box(s,a,x,y,text,'state')
  s.arrow(a,[(x,inp['bottom']),(x,350),(bus,350),(bus,y),(n['left'],y)])
  cy=y+105;radius=13
  a.add_patch(Circle((x,cy),radius,facecolor='white',edgecolor=s.cfg['colors']['ink'],linewidth=s.cfg['box']['stroke_width'],zorder=3))
  s.text(a,x,cy,'+',role='heading')
  s.arrow(a,[(x,n['bottom']),(x,cy-radius)])
  s.arrow(a,[(bus+15,y),(bus+15,cy),(x-radius,cy)])
  s.text(a,x-90,cy-23,alpha,role='formula')
  results.append((x+radius,cy))
 end=box(s,a,x,1050,'Concat → 1×1：256→256\n产生 '+('L' if x==200 else 'H'),'output');fused.append(end)
 for px,py in results:s.arrow(a,[(px,py),(collector,py),(collector,980),(x,980),(x,end['top'])])
 gate=box(s,a,x,1230,(r'$G_L=\mathrm{Gate}(L,H)$' if x==200 else r'$G_H=\mathrm{Gate}(H,L)$')+'\n'+('H 调 L' if x==200 else 'L 调 H'),'projection');gated.append(gate);edge(s,a,end,gate)
control(s,a,[(fused[1]['x'],fused[1]['bottom']),(gated[0]['x'],gated[0]['top'])])
control(s,a,[(fused[0]['x'],fused[0]['bottom']),(gated[1]['x'],gated[1]['top'])])
end=box(s,a,400,1420,'Concat：256＋256＝512\nDW3(512通道) → PW1(512→256)\nM(U)：[B,256,8,8]','output')
for n in gated:s.arrow(a,[(n['x'],n['bottom']),(n['x'],1320),(400,1320),(400,end['top'])])
s.text(a,400,1540,r'每支：$F_i(X_i)+\alpha X_i$；同频三支共用一个α，初值1。',role='annotation')
s.text(a,400,1580,'无同路循环门控，无Attention；RepDW内部恒等分支仍保留。',role='annotation')
save(s,f,'E2_1_02_分频分支与跨频融合')

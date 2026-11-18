"""E1 single reading figure. Documentation only; imports shared renderer read-only."""
from pathlib import Path
import sys, json, hashlib, subprocess
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
sys.path.insert(0,str(ROOT/'source/analysis/tinyvim_guide'))
from draw_block_details import Diagram
OUT=ROOT/'研究记录/模块设计/figures/三模块'
class Main(Diagram):
 def __init__(self):
  from figure_style import FigureStyle
  self.s=FigureStyle();self.s.cfg['output']['width_px']=1700
  self.s.cfg['sizes'].update(heading=23,annotation=20,dimension=20,formula=24)
  self.s.cfg['box'].update(padding_x=13,padding_y=10,row_gap=6)
  self.fig,self.ax=self.s.canvas(1480);self.nodes={}
 def box(self,key,x,y,text,color='projection',role='annotation'):
  n=self.s.box(self.ax,x,y,text,self.s.cfg['colors'][color],role=role);self.nodes[key]=n;return n
 def text(self,x,y,t,role='annotation'):self.s.text(self.ax,x,y,t,role=role)
 def par(self,a,b,via=None,start='bottom',end='top'):
  from matplotlib.patches import FancyArrowPatch
  n,m=self.nodes[a],self.nodes[b]
  def port(n,p):return (n['x'],n[p]) if p in ('top','bottom') else (n[p],n['y'])
  pts=[port(n,start)]+(via or [])+[port(m,end)]
  for p,q in zip(pts[:-2],pts[1:-1]):self.ax.plot([p[0],q[0]],[p[1],q[1]],color='#4278a0',lw=1.5,ls='--',zorder=1)
  self.ax.add_patch(FancyArrowPatch(pts[-2],pts[-1],arrowstyle='-|>',mutation_scale=12,color='#4278a0',lw=1.5,linestyle='--',zorder=2))
d=Main()
d.text(850,35,'E1｜在最深解码层进行方向性波传播',role='heading')
d.text(295,90,'整网位置：只替换 D3 ×1',role='heading')
d.text(1160,90,'D3 展开：输入 X → 频域传播 → 门控 → 残差',role='heading')
# Whole network: four levels and all three skips.
d.box('rgb',110,165,'RGB 输入\n3×256×256','input')
d.box('stem',110,260,'Stem\n48×64×64');d.link('rgb','stem')
ys=[400,640,880,1120];cs=[48,64,168,224];ac=[64,128,256,512]
for i,y in enumerate(ys):
 z=64//2**i
 d.box(f's{i}',110,y,f'Stage {i}\n{cs[i]}×{z}×{z}','input')
 d.box(f'a{i}',295,y,f'对齐\n{ac[i]}通道')
 d.link(f's{i}',f'a{i}',start='right',end='left')
 if i<3:
  d.box(f'down{i}',110,y+120,'下采样×2');d.link(f's{i}',f'down{i}')
 if i>0:d.link(f'down{i-1}',f's{i}')
 d.box(f'd{i}',485,y,('D3：D-WPO\n替换×1' if i==3 else f'D{i}\n拼接 + 双卷积'),'state' if i==3 else 'output')
 d.link(f'a{i}',f'd{i}',start='right',end='left')
 if i<3:
  d.box(f'up{i}',485,y+120,'上采样×2');d.link(f'up{i}',f'd{i}',start='top',end='bottom')
for i in range(3):d.link(f'd{i+1}',f'up{i}',start='top',end='bottom')
d.link('stem','s0')
d.box('head',485,245,'输出头 + 恢复尺寸\nSigmoid\n1×256×256','output');d.link('d0','head',start='top',end='bottom')
d.text(295,1280,'S0 / S1 / S2 三条 skip 保留\nD2 / D1 / D0 及编码器均保留\n尺寸省略批量 B：通道×高×宽')
# Expanded D3, one continuous causal graph.
d.box('x',765,165,'输入 X\n512×8×8','input')
d.box('dw',1020,165,'DW 3×3\n逐通道卷积')
d.box('proj',1390,165,'Linear：512→1536\n拆为 U0 / V0 / Z')
d.link('x','dw',start='right',end='left');d.link('dw','proj',start='right',end='left')
d.par('d3','x',[(620,1120),(620,165)],start='right',end='left')
d.text(615,650,'展\n开')
d.box('fft',1000,330,'U0 / V0 → 正交 rFFT2\n场频谱 F / 速度频谱 G\n各为 512×8×5','input')
d.link('proj','fft',[(1390,245),(1000,245)])
d.box('rho',1440,335,r'$\rho^2=a_x k_x^2+a_y k_y^2$'+'\n'+r'$a_x=1+\tanh(a),\ a_y=1-\tanh(a)$'+'\nkx / ky：水平 / 垂直频率；a 初始0','state',None)
d.box('q',1425,515,r'$q=(\rho^2)^p,\quad\omega=\sqrt{q}$'+'\np=softplus(p_raw+p_ch)，初始1','state',None)
d.par('rho','q')
d.box('alpha',885,510,'阻尼 α = softplus(A_grid + A_ch)\n下限 0.0001；网格初始0.05\n通道偏置小扰动；固定 t=1','projection')
d.box('damp',900,740,'阻尼响应 D\n'+r'$D=e^{-\alpha q t}F$','state',None)
d.box('osc',1400,750,'振荡响应 O\n'+r'$O=e^{-\alpha qt/2}[F\cos(\omega t)$'+'\n'+r'$+(G+\alpha qF/2)\frac{\sin(\omega t)}{\omega}]$','state',None)
# Solid feature dependencies; dashed coefficient dependencies.
d.link('fft','damp',[(680,330),(680,620),(900,620)],start='left')
d.link('fft','osc',[(1190,330),(1190,625),(1400,625)],start='right')
d.par('alpha','damp')
d.par('alpha','osc',[(885,595),(1645,595),(1645,750)],end='right')
d.par('q','osc')
d.par('q','damp',[(1425,605),(1100,605),(1100,740)],end='right')
d.box('mix',1160,950,r'$F_{out}=D+w(O-D)$'+'\nw：可学习混合权重，初始0','state',None)
d.link('damp','mix',[(900,855),(1160,855)])
d.link('osc','mix',[(1400,855),(1160,855)])
d.box('ifft',1450,1110,'正交 irFFT2 → LN\n恢复 512×8×8')
d.link('mix','ifft',[(1450,950)])
d.box('gate',1160,1110,'逐元素乘\nSiLU(Z)','state')
d.box('linear',850,1110,'输出 Linear\n512→512')
d.link('ifft','gate',start='left',end='right');d.link('gate','linear',start='left',end='right')
d.link('proj','gate',[(1680,165),(1680,1205),(1160,1205)],start='right',end='bottom')
d.text(1590,1205,'门控 Z')
d.box('gamma',850,1305,'乘逐通道 γ\n初始0.001','state')
d.box('add',1160,1305,r'$Y=X+\gamma\odot wave(X)$','output',None)
d.box('y',1470,1305,'输出 Y\n→ 原上采样、D2','output')
d.link('linear','gamma');d.link('gamma','add',start='right',end='left');d.link('add','y',start='right',end='left')
d.link('x','add',[(655,165),(655,1400),(1160,1400)],start='left',end='bottom')
d.text(850,1400,'保留原输入 X（恒等残差）')
d.text(1165,1450,'实线：特征；蓝色虚线：系数 / 位置展开。数值保护与参数细节见正文备查。')
name='E1_主图_整网位置与完整计算'
audit=d.s.save(d.fig,OUT/name)
assert audit['passed'],audit['issues']
for w,suffix in [(1100,'.png'),(1100,'_1100.png'),(800,'_800.png'),(800,'_preview800.png')]:
 subprocess.run(['inkscape',str(OUT/(name+'.svg')),'--export-type=png',f'--export-width={w}','--export-filename='+str(OUT/(name+suffix))],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
print(json.dumps({'svg':str(OUT/(name+'.svg')),'sha256':hashlib.sha256((OUT/(name+'.svg')).read_bytes()).hexdigest(),'audit':audit['passed']},ensure_ascii=False))

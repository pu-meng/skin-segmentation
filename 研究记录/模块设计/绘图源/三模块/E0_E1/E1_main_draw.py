"""E1 integrated paper figure; current candidate.py. Documentation only.
Reuses existing Main/Diagram renderer; no experiment code changes.
"""
from pathlib import Path
import sys,json,hashlib,subprocess
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
sys.path.insert(0,str(ROOT/'source/analysis/tinyvim_guide'))
from draw_block_details import Diagram
OUT=ROOT/'研究记录/模块设计/figures/三模块'
class Main(Diagram):
 def __init__(self):
  from figure_style import FigureStyle
  self.s=FigureStyle();self.s.cfg['output']['width_px']=1700
  self.s.cfg['sizes'].update(heading=23,annotation=20,dimension=20,formula=24)
  self.s.cfg['box'].update(padding_x=12,padding_y=10,row_gap=5)
  self.fig,self.ax=self.s.canvas(1400);self.nodes={}
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
 def panel(self,x,y,w,h):
  from matplotlib.patches import FancyBboxPatch
  self.ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=12',facecolor='#fafbfd',edgecolor='#b8c7d4',linewidth=1,zorder=0))
d=Main()
d.text(850,35,'E1｜在最深解码层进行方向性波传播','heading')
d.panel(12,76,545,780);d.panel(580,76,1108,780);d.panel(12,880,1676,495)
d.text(285,108,'(a) 四层整网：仅替换 D3 ×1','heading')
d.text(1130,108,'(b) DStage：完整方向性波传播模块','heading')
d.box('rgb',105,175,'RGB 输入\n3×256×256','input')
d.box('stem',105,270,'Stem ×4');d.link('rgb','stem')
d.box('head',440,215,'输出头 → ×4\nSigmoid\n1×256×256','output')
ys=[370,500,630,760];cs=[48,64,168,224];ac=[64,128,256,512]
for i,y in enumerate(ys):
 z=64//2**i
 d.box(f's{i}',105,y,f'Stage {i}\n{cs[i]}×{z}²','input')
 d.box(f'a{i}',265,y,f'对齐\n{ac[i]}通道')
 d.box(f'd{i}',440,y,('D3：DWPO\n512×8²' if i==3 else f'D{i}：拼接\n双卷积 → {ac[i]}'),'state' if i==3 else 'output')
 d.link(f's{i}',f'a{i}',start='right',end='left');d.link(f'a{i}',f'd{i}',start='right',end='left')
 if i>0:
  d.link(f's{i-1}',f's{i}');d.link(f'd{i}',f'd{i-1}',start='top',end='bottom')
  d.text(157,y-65,'×2');d.text(485,y-65,'×2')
d.link('stem','s0');d.link('d0','head',start='top',end='bottom')
d.text(280,820,'三条 skip 保留；尺寸为 C×H×W')
d.box('x',680,190,'输入 X\n512×8×8','input')
d.box('dw',920,190,'DW 3×3\n逐通道卷积')
d.box('proj',1280,190,'Linear：512→1536\n拆分 U0、V0、Z')
d.link('x','dw',start='right',end='left');d.link('dw','proj',start='right',end='left')
d.par('d3','x',[(566,760),(566,190)],start='right',end='left')
d.box('fft',920,350,'U0、V0 → rFFT2\n场频谱 F、速度频谱 G\n各为 512×8×5','input')
d.box('prop',1330,350,'方向性频域传播\nF、G → Fout\n计算展开于 (c)','state')
d.link('proj','fft',[(1280,270),(920,270)]);d.link('fft','prop',start='right',end='left')
d.box('ifft',1500,510,'irFFT2\n512×8×8')
d.box('ln',1280,510,'LN\n按通道归一化')
d.box('gate',1020,510,'⊙ SiLU(Z)','state')
d.box('linear',760,510,'输出 Linear\n512→512')
d.link('prop','ifft',[(1500,350)])
d.link('ifft','ln',start='left',end='right');d.link('ln','gate',start='left',end='right');d.link('gate','linear',start='left',end='right')
d.link('proj','gate',[(1635,190),(1635,600),(1020,600)],start='right',end='bottom')
d.text(1485,575,'门控支路 Z')
d.box('gamma',760,690,'乘逐通道 γ\n初始 0.001','state')
d.box('add',1110,690,'＋ X','output')
d.box('y',1450,690,'输出 Y：512×8×8\n→ 上采样、D2','output')
d.link('linear','gamma');d.link('gamma','add',start='right',end='left');d.link('add','y',start='right',end='left')
d.link('x','add',[(602,190),(602,782),(1110,782)],start='left',end='bottom')
d.text(850,755,'恒等残差 X')
d.text(1130,825,'FP32 传播；A 在 rFFT 边界对称化后记为 Ā')
d.text(850,915,'(c) 方向频率 → 阻尼与振荡 → 混合频谱（单头，固定 t=1）','heading')
d.box('freq',330,1003,r'$k_x=2\pi\,\mathrm{rfftfreq}(W),\ k_y=2\pi\,\mathrm{fftfreq}(H)$'+'\n'+r'$\lambda_c=(1+\tanh a_c)k_x^2+(1-\tanh a_c)k_y^2$','state',role='formula')
d.box('q',330,1120,r'$p_c=\mathrm{softplus}(p_{raw}+p_{ch,c})$'+'\n'+r'$q_c=\exp[p_c\log(\max(\lambda_c,10^{-12}))]$','state',role='formula')
d.link('freq','q')
d.box('alpha',330,1255,r'$\alpha=\max[\mathrm{softplus}(\bar A+A_{ch}),10^{-4}]$'+'\n'+r'$\omega=\sqrt{\max(q,10^{-12})},\ \omega_s=\max(\omega,10^{-6})$','projection',role='formula')
d.box('damp',1050,1005,'阻尼 D\n'+r'$D=e^{-\alpha qt}F$','state',role=None)
d.box('osc',1290,1140,'振荡 O\n'+r'$O=e^{-\alpha qt/2}[F\cos(\omega t)$'+'\n'+r'$+(G+\alpha qF/2)\sin(\omega t)/\omega_s]$','state',role=None)
d.box('mix',1240,1300,r'$F_{out}=D+w(O-D)$'+'\nw：可学习混合权重，初始 0','output',role=None)
d.par('q','damp',[(700,1120),(700,1005)],start='right',end='left')
d.par('alpha','osc',[(700,1255),(700,1140)],start='right',end='left')
d.ax.plot([700,700],[1120,1140],color='#4278a0',lw=1.5,ls='--',zorder=1)
d.text(790,958,'α、q、ω；t=1')
d.link('damp','mix',[(1050,1070),(870,1070),(870,1300)],end='left');d.link('osc','mix')
d.par('prop','osc',[(1665,350),(1665,1140)],start='right',end='right')
d.text(1585,950,'F、G')
d.text(400,1350,'a 初始0；p 初始1；c 为通道下标')
d.text(1225,1360,'FFT 正交归一化；实线：特征；虚线：系数／展开')
name='E1_主图_整网位置与完整计算'
audit=d.s.save(d.fig,OUT/name)
assert audit['passed'],audit['issues']
for w,suffix in [(1100,'.png'),(1100,'_1100.png'),(800,'_800.png'),(800,'_preview800.png')]:
 subprocess.run(['inkscape',str(OUT/(name+'.svg')),'--export-type=png',f'--export-width={w}','--export-filename='+str(OUT/(name+suffix))],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
print(json.dumps({'svg':str(OUT/(name+'.svg')),'sha256':hashlib.sha256((OUT/(name+'.svg')).read_bytes()).hexdigest(),'audit':audit['passed']},ensure_ascii=False))

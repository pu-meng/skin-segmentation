"""Compact E1: actual WaveFormer FFT implementation, not WaveR DCT."""
from pathlib import Path
import sys
sys.path.insert(0,'/home/PuMengYu/.codex/skills/research-figure-style/scripts')
from figure_style import FigureStyle

def draw():
 s=FigureStyle();s.cfg['sizes'].update(heading=17,formula=19,dimension=16,annotation=15)
 s.cfg['box'].update(padding_x=16,padding_y=11,row_gap=6)
 fig,ax=s.canvas(865);nodes={}
 def text(x,y,t):s.text(ax,x,y,t,role='annotation')
 def n(k,x,y,t,c='projection'):
  nodes[k]=s.box(ax,x,y,t,s.cfg['colors'][c],role=None)
 def edge(a,b,side='bottom',target='top',route=None):
  a,b=nodes[a],nodes[b]
  def port(n,p):return (n['x'],n[p]) if p in ('top','bottom') else (n[p],n['y'])
  p,q=port(a,side),port(b,target)
  s.arrow(ax,route or [p,q])
 s.text(ax,400,27,'E1 · 编码器–解码器过渡处的方向性波传播',role='heading')
 n('enc',135,112,'TinyViM 编码器\n原通道对齐','input')
 n('d3',400,112,'最深 D3 替换 ×1\n512通道 · 8×8','state')
 n('dec',665,112,'D2 → D1 → D0\n原输出头','output')
 edge('enc','d3','right','left');edge('d3','dec','right','left')
 text(660,181,'上采样与三条 Skip 保持')
 ax.plot([400,400],[nodes['d3']['bottom'],211],ls='--',color=s.cfg['colors']['ink'],lw=1.2)
 n('x',380,255,'D3 输入 X · [B,512,8,8]','input')
 n('project',380,355,'DW3 → Linear\n拆成 U0、V0、Z')
 n('fft',380,465,'U0、V0 → rFFT2\n得到频谱 F0、G0')
 n('core',380,565,'方向 WPO 核心\n同一传播函数，只替换频率项 q','state')
 n('inverse',380,675,'irFFT2 → LayerNorm\n× SiLU(Z) 内容门控')
 n('out',380,785,'输出 Linear → γ 缩放 → +X\nD3 输出 Y · 交给 D2','output')
 for a,b in [('x','project'),('project','fft'),('fft','core'),('core','inverse'),('inverse','out')]:edge(a,b)
 a,b=nodes['project'],nodes['inverse'];edge('project','inverse','right','right',[(a['right'],a['y']),(720,a['y']),(720,b['y']),(b['right'],b['y'])])
 text(690,490,'Z');text(681,525,'旁路')
 a,b=nodes['x'],nodes['out'];edge('x','out','left','left',[(a['left'],a['y']),(50,a['y']),(50,b['y']),(b['left'],b['y'])])
 text(101,565,'原始 X')
 text(400,841,'γ 初值 0.001；无新增 FFN；两条频率公式见正文')
 return s.save(fig,Path(__file__).parent/'三模块/E1_简图_位置与WPO')
if __name__=='__main__': print(draw())

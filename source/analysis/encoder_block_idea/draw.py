"""Six-branch candidate; no training or model implementation."""
from pathlib import Path
import sys
sys.path.insert(0, '/home/PuMengYu/.codex/skills/research-figure-style/scripts')
from figure_style import FigureStyle
OUT=Path('/home/PuMengYu/TinyViM_Encoder_Block_Idea_figures')
s=FigureStyle();f,a=s.canvas(1600)
s.text(a,400,30,'Encoder Block：六条分支与三类门控',role='heading')
def box(x,y,text,color):
    s.measure(a,text)
    return s.box(a,x,y,text,s.cfg['colors'][color])
def route(p,q,y):
    s.arrow(a,[(p['x'],p['bottom']),(p['x'],y),(q['x'],y),(q['x'],q['top'])])
x=box(400,100,'输入特征 X','input')
locals=[box(pos,265,f'局部分支{i+1}\nDWConv {k}×{k}','projection') for i,(pos,k) in enumerate(zip([150,400,650],[3,5,7]))]
g1=box(400,460,'门控①：局部尺度选择\n读取三路局部特征\n按位置调整三种尺度的贡献','state')
g2=box(400,695,'门控②：局部引导全局\n局部特征 → 三张空间门控\n分别调节原始 X 的三路副本\n作用于 FFT 之前','state')
globals=[box(pos,950,f'全局分支{i+1}：{band}\nFFT → 频域混合\n→ IFFT','projection') for i,(pos,band) in enumerate(zip([150,400,650],['低频','高频','全频']))]
gm=box(400,1140,'三路全局特征汇合','output')
g3=box(400,1320,'门控③：局部—全局融合\n读取局部特征与全局特征\n调整两类信息对输出的贡献','state')
y=box(400,1495,'输出特征 Y','output')
for n in locals:route(x,n,180);route(n,g1,355)
route(g1,g2,575)
for n in globals:route(g2,n,835);route(n,gm,1060)
route(gm,g3,1225);route(g3,y,1425)
s.arrow(a,[(x['left'],x['y']),(24,x['y']),(24,g2['y']),(g2['left'],g2['y'])])
s.text(a,89,650,'原始 X',role='annotation')
s.arrow(a,[(g1['right'],g1['y']),(780,g1['y']),(780,g3['y']),(g3['right'],g3['y'])])
s.text(a,708,1260,'局部特征',role='annotation')
s.text(a,400,1568,'候选设计；频带划分、门控公式及效果待验证。',role='annotation')
s.save(f,OUT/'03_Encoder_Block_概念结构')

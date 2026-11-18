"""Two-stage skip1 Generator, in the operator style of the reference guide."""
from pathlib import Path
import sys
sys.path.insert(0,'/home/PuMengYu/.codex/skills/research-figure-style/scripts')
from figure_style import FigureStyle
OUT=Path(__file__).resolve().parent


def draw():
    s=FigureStyle();fig,ax=s.canvas(1730);c=s.cfg['colors']
    def box(x,y,text,color='projection'):return s.box(ax,x,y,text,c[color])
    def down(a,b):s.arrow(ax,[(a['x'],a['bottom']),(b['x'],b['top'])])
    def fan(a,children):
        y=a['bottom']+20
        ax.plot([a['x'],a['x']],[a['bottom'],y],color=c['ink'],lw=1.25)
        ax.plot([children[0]['x'],children[-1]['x']],[y,y],color=c['ink'],lw=1.25)
        for node in children:s.arrow(ax,[(node['x'],y),(node['x'],node['top'])])
    def join(nodes,dest):
        for n,port in zip(nodes,[dest['left']+30,dest['x'],dest['right']-30]):
            s.arrow(ax,[(n['x'],n['bottom']),(n['x'],dest['top']-20),(port,dest['top']-20),(port,dest['top'])])
    s.text(ax,400,28,'skip1 · Generator',role='heading')
    z=box(400,100,'输入 [B,64,64,64]','input')
    split=box(400,190,'按通道拆分：24／24／16','input');down(z,split)
    xs=[140,400,660];entries=[]
    for x,text in zip(xs,['Laplacian · 24 ch','Haar · 24 ch','Fourier · 16 ch']):entries.append(box(x,285,text,'input'))
    fan(split,entries)
    rows=[
        ['LN → t','LN','LN'],
        ['5×5高斯平滑','Haar分解\n24→96；32×32','rFFT2'],
        ['AvgPool ×2\n插值回64×64','DW3 → GELU\n1×1：96→96','× 可学习掩码m'],
        ['t − 低频结果','Haar重建\n96→24；64×64','irFFT2'],
        ['DW3 → GELU → PW\n＋本路输入','＋本路输入','GELU → PW\n＋本路输入'],
    ]
    previous=entries
    for y,texts in zip([375,470,580,690,800],rows):
        current=[]
        for x,text,prev in zip(xs,texts,previous):
            n=box(x,y,text);down(prev,n);current.append(n)
        previous=current
    mix=box(400,915,'Concat → 1×1(64→64) → LN','output');join(previous,mix)
    split2=box(400,1005,'再按通道拆分：24／24／16','input');down(mix,split2)
    second=[]
    for x,text in zip(xs,[
        'LN → RepDW3\nGELU → PW\n＋本路输入',
        'LN，24→12＋12\nDW1×7／DW7×1\nConcat → GELU → PW\n＋本路输入',
        'AvgPool ×4 → LN\nRepDW7 → GELU\nPW → 插值回64×64\n＋本路输入']):
        second.append(box(x,1170,text,'state'))
    fan(split2,second)
    features=[]
    for x,text,prev in zip(xs,['a · 24 ch','b · 24 ch','c · 16 ch'],second):
        n=box(x,1325,text,'output');down(prev,n);features.append(n)
    joined=box(400,1420,'Concat(a,b,c)：64通道','input');join(features,joined)
    gate=box(400,1530,'GAP → MLP64→16→3 → Softmax\nw [B,3,1,1]；初始各1/3','state');down(joined,gate)
    final=box(400,1655,'$\\mathrm{PW}(\\mathrm{Concat}(3w_0a,3w_1b,3w_2c))$\n输出 [B,64,64,64]','output');down(gate,final)
    s.compact_vertical(fig);s.save(fig,OUT/'DCG_内容生成详细')


if __name__=='__main__':draw()

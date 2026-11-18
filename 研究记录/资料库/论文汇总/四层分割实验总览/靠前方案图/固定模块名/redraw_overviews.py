"""A0/A1 four-stage U networks, matching the established TinyViM guide layout."""
from pathlib import Path
import sys
sys.path.insert(0,'/home/PuMengYu/.codex/skills/research-figure-style/scripts')
from figure_style import FigureStyle
from matplotlib.patches import Circle

OUT=Path(__file__).resolve().parent


def draw(variant):
    s=FigureStyle();fig,ax=s.canvas(1580);c=s.cfg['colors'];nodes={}
    left,mid,join,right=145,355,510,665
    def box(key,x,y,text,color):
        n=s.box(ax,x,y,text,c[color]);nodes[key]=n;return n
    def vertical(a,b):
        a,b=nodes[a],nodes[b]
        p=(a['x'],a['bottom'] if a['y']<b['y'] else a['top'])
        q=(b['x'],b['top'] if a['y']<b['y'] else b['bottom'])
        s.arrow(ax,[p,q])
    def horizontal(a,b):
        a,b=nodes[a],nodes[b];s.arrow(ax,[(a['right'],a['y']),(b['left'],b['y'])])
    s.text(ax,400,28,variant+' · TinyViM-S＋四层Decoder',role='heading')
    s.text(ax,left,72,'TinyViM-S 编码器',role='annotation')
    s.text(ax,right,72,'四层Decoder',role='annotation')
    box('input',left,155,'RGB 输入\n3 ch · 256×256\nImageNet均值／方差归一化','input')
    box('stem',left,340,'Stem\n3×3，步长2：3→24\n3×3，步长2：24→48\n每层BN＋GELU','projection');vertical('input','stem')
    ys=[610,850,1090,1330];chs=[48,64,168,224];aligned=[64,128,256,512]
    seqs=['Local ×2 → TViM ×1','Local ×2 → TViM ×1',
          'Local ×4 → TViM ×1\nLocal ×3 → TViM ×1','Local ×5 → TViM ×1']
    for i,(y,ch,co,seq) in enumerate(zip(ys,chs,aligned,seqs)):
        name=f'Stage {i}'+(' → BN' if i==3 else '')
        box(f'stage{i}',left,y,f'{name}\n{seq}\n{ch} ch · {64//2**i}×{64//2**i}','input')
        box(f'align{i}',mid,y,f'1×1投影\n{ch}→{co}','projection');horizontal(f'stage{i}',f'align{i}')
        if i<3:
            box(f'down{i}',left,y+120,f'下采样 ×2\n3×3，步长2＋BN\n{ch}→{chs[i+1]}','projection')
            vertical(f'stage{i}',f'down{i}')
    vertical('stem','stage0')
    for i in range(3):vertical(f'down{i}',f'stage{i+1}')
    # D0 is above the shallow fusion to leave room for processing both inputs.
    box('d0',right,420,'D0 · DoubleConv\n128→64；64×64','output')
    box('head',right,205,'1×1：64→1\n双线性上采样 ×4\nSigmoid；256×256','output');vertical('d0','head')
    box('skip1',540,610,'skip1＋Concat\n64＋64→128\n64×64','state');horizontal('align0','skip1')
    box('up0',right,730,'上采样 ×2\n128→64','state')
    a,b=nodes['up0'],nodes['skip1']
    s.arrow(ax,[(right,a['top']),(right,675),(b['x'],675),(b['x'],b['bottom'])])
    a,b=nodes['skip1'],nodes['d0']
    s.arrow(ax,[(a['x'],a['top']),(a['x'],510),(right,510),(right,b['bottom'])])
    for i in [1,2]:
        y=ys[i];ch=aligned[i];size=64//2**i
        name='D1 · DoubleConv' if i==1 else ('D2 · 双分支' if variant=='A1' else 'D2 · DoubleConv')
        box(f'd{i}',right,y,f'{name}\n{2*ch}→{ch}；{size}×{size}',
            'state' if variant=='A1' and i==2 else 'output')
        box(f'up{i}',right,y+120,f'上采样 ×2\n{2*ch}→{ch}','state')
        ax.add_patch(Circle((join,y),14,facecolor='white',edgecolor=c['ink'],lw=1.1,zorder=3))
        s.text(ax,join,y,'C',role='annotation')
        a,b,u=nodes[f'align{i}'],nodes[f'd{i}'],nodes[f'up{i}']
        s.arrow(ax,[(a['right'],y),(join-14,y)])
        s.arrow(ax,[(join+14,y),(b['left'],y)])
        elbow=(u['top']+b['bottom'])/2
        s.arrow(ax,[(right,u['top']),(right,elbow),(join,elbow),(join,y+14)])
    vertical('d1','up0');vertical('d2','up1')
    bottle=box('bottle',mid,1470,'bottleneck\n512 ch · 8×8','state');vertical('align3','bottle')
    wave=box('d3',right,1470,'D3 · D-WPO\n512→512；8×8','state');horizontal('bottle','d3');vertical('d3','up2')
    s.text(ax,mid,557,'S0 · 64×64',role='annotation')
    s.text(ax,mid,ys[1]-55,'S1 · 32×32',role='annotation')
    s.text(ax,mid,ys[2]-55,'S2 · 16×16',role='annotation')
    s.text(ax,400,1550,'C：通道拼接；D3 → D2 → D1 → D0',role='annotation')
    s.compact_vertical(fig);s.save(fig,OUT/(variant+'_位置'))


if __name__=='__main__':draw('A0');draw('A1')

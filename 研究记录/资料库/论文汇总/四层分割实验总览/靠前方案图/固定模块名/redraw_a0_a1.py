"""Only A0 skip1 interaction and A1 D2; operators instead of prose blocks."""
from pathlib import Path
import sys
sys.path.insert(0,'/home/PuMengYu/.codex/skills/research-figure-style/scripts')
from figure_style import FigureStyle
from matplotlib.patches import Circle

OUT=Path(__file__).resolve().parent


def setup(height):
    s=FigureStyle();fig,ax=s.canvas(height);c=s.cfg['colors']
    def box(x,y,text,color='input'):return s.box(ax,x,y,text,c[color])
    def down(a,b):s.arrow(ax,[(a['x'],a['bottom']),(b['x'],b['top'])])
    def right(a,b):s.arrow(ax,[(a['right'],a['y']),(b['left'],b['y'])])
    def circle(x,y,symbol):
        ax.add_patch(Circle((x,y),17,facecolor=c['output'],edgecolor=c['ink'],lw=1.1,zorder=3))
        s.text(ax,x,y,symbol,role='formula')
        return dict(x=x,y=y,left=x-17,right=x+17,top=y-17,bottom=y+17)
    return s,fig,ax,box,down,right,circle


def skip():
    s,fig,ax,box,down,right,circle=setup(1730)
    s.text(ax,400,30,'A0 · skip1',role='heading')
    for x,name,u,content,color in [(200,'S','u_s','k','projection'),(600,'D','u_d','v','state')]:
        entry=box(x,100,name+' [B,64,64,64]',color)
        pre=box(x,205,'DW3 → 1×1\nLN → GELU',color);down(entry,pre)
        port=box(x,310,'$'+u+'$',color);down(pre,port)
        gen=box(x,405,'LN → Generator',color);down(port,gen)
        out=box(x,490,'$'+content+'$',color);down(gen,out)
    # Named k/v ports keep the two control directions separate; no crossing wires.
    for y,label,gate_input,content,output,first_color,content_color in [
        (710,'S侧', 'v','k','k_h','state','projection'),
        (920,'D侧', 'k','v','v_h','projection','state')]:
        s.text(ax,100,y-90,label,role='heading')
        source=box(55,y,'$'+gate_input+'$',first_color)
        gap=box(175,y,'GAP',first_color);right(source,gap)
        mlp=box(365,y,'MLP\n64→16→64',first_color);right(gap,mlp)
        sigmoid=box(535,y,'$2\\sigma$',first_color);right(mlp,sigmoid)
        product=circle(655,y,'$\\times$');right(sigmoid,product)
        raw=box(655,y-90,'$'+content+'$',content_color);down(raw,product)
        output=box(750,y,'$'+output+'$','output');right(product,output)
    s.text(ax,400,1030,'两侧回写',role='heading')
    for x,kh,u,h,t,out,entry,color in [
        (200,'k_h','u_s','h_s','t_s',"S'",'S','projection'),
        (600,'v_h','u_d','h_d','t_d',"D'",'D','state')]:
        p=box(x,1100,'$'+kh+'$','output')
        a=box(x,1200,'$'+h+'='+u+'+\\mathrm{PW}('+kh+')$',color);down(p,a)
        b=box(x,1320,'$'+t+'='+h+'+$\n$\\mathrm{FFN}(\\mathrm{LN}('+h+'))$',color);down(a,b)
        gamma='\\gamma_s' if entry=='S' else '\\gamma_d'
        d=box(x,1460,'$'+out+'='+entry+'+'+gamma+'\\mathrm{Post}('+t+')$','output');down(b,d)
        if entry=='S':left=d
        else:right_node=d
    joined=box(400,1580,"Concat(D′, S′) → D0",'output')
    for node,port in [(left,joined['left']+35),(right_node,joined['right']-35)]:
        s.arrow(ax,[(node['x'],node['bottom']),(node['x'],joined['top']-20),
                    (port,joined['top']-20),(port,joined['top'])])
    s.text(ax,400,1680,'Post：DW3 → 1×1 → LN → GELU\nγs、γd初值0.01；两侧参数独立',role='annotation')
    s.compact_vertical(fig);s.save(fig,OUT/'DCG_双源交互')


def d2():
    s,fig,ax,box,down,right,circle=setup(1130)
    s.text(ax,400,30,'A1 · D2',role='heading')
    x=box(400,100,'输入 [B,512,16,16]')
    project=box(400,190,'1×1：512→256');down(x,project)
    u=box(400,275,'U [B,256,16,16]');down(project,u)
    left=box(200,410,'LN → Mamba-1\n展平空间，再还原二维','projection')
    local=box(600,410,'LN → RepDW7\nGELU → 1×1(256→256)','state')
    y=u['bottom']+25;c=s.cfg['colors']['ink']
    ax.plot([400,400],[u['bottom'],y],color=c,lw=1.25)
    ax.plot([200,600],[y,y],color=c,lw=1.25)
    for n in [left,local]:s.arrow(ax,[(n['x'],y),(n['x'],n['top'])])
    for x,node,symbol,weight,color in [(200,left,'M','\\alpha','projection'),(600,local,'C','\\beta','state')]:
        branch=box(x,535,'$'+symbol+'$',color);down(node,branch)
        product=circle(x,630,'$\\times$');down(branch,product)
        px=80 if x==200 else 720
        parameter=box(px,630,'$'+weight+'$','state')
        if x==200:right(parameter,product)
        else:s.arrow(ax,[(parameter['left'],630),(product['right'],630)])
        weighted=box(x,720,'$'+weight+' '+symbol+'$','output');down(product,weighted)
        if x==200:a=weighted
        else:b=weighted
    original=box(400,790,'U','input')
    add=box(400,890,'$Y=U+\\alpha M+\\beta C$','output');down(original,add)
    for n,p in [(a,add['left']+30),(b,add['right']-30)]:
        s.arrow(ax,[(n['x'],n['bottom']),(n['x'],add['top']-20),(p,add['top']-20),(p,add['top'])])
    out=box(400,990,'Y [B,256,16,16]','output');down(add,out)
    s.text(ax,400,1080,'$\\alpha=0.1\\tanh(\\mathrm{raw}_m),\\quad\\beta=0.1\\tanh(\\mathrm{raw}_r)$\nα、β为可学习标量，初值均为0.01',role='annotation')
    s.compact_vertical(fig);s.save(fig,OUT/'A1_D2')


if __name__=='__main__':skip();d2()

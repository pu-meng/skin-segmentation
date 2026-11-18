"""A0 bottleneck: unambiguous one-way flows, shared measured figure style."""
from pathlib import Path
import sys
sys.path.insert(0,'/home/PuMengYu/.codex/skills/research-figure-style/scripts')
from figure_style import FigureStyle

OUT=Path(__file__).resolve().parent


def main():
    s=FigureStyle();fig,ax=s.canvas(1510);c=s.cfg['colors']
    def box(x,y,text,color):return s.box(ax,x,y,text,c[color])
    def down(a,b):s.arrow(ax,[(a['x'],a['bottom']),(b['x'],b['top'])])
    s.text(ax,400,30,'A0 · bottleneck：三层Mamba与频谱通道加权',role='heading')
    x=box(400,85,'X [B,512,8,8]','input')
    pre=box(400,170,'DW3 → 1×1(512→256) → LN → GELU','input');down(x,pre)
    u=box(400,265,'U [B,256,8,8]\n沿通道均分：Us、Uc各128通道','input');down(pre,u)
    us=box(200,375,'Us [B,128,8,8]','projection')
    uc=box(600,375,'Uc [B,128,8,8]','state')
    # One stem, a headless split bar, and two downward arrowheads.
    joint=u['bottom']+24
    ax.plot([400,400],[u['bottom'],joint],color=c['ink'],lw=1.25)
    ax.plot([200,600],[joint,joint],color=c['ink'],lw=1.25)
    for node in [us,uc]:s.arrow(ax,[(node['x'],joint),(node['x'],node['top'])])
    prev=us
    for i,y in enumerate([465,565,665],1):
        node=box(200,y,f'单向 Mamba Block {i}\nLN → Mamba → 加回输入','projection');down(prev,node);prev=node
    a=box(600,475,'从Uc的频谱统计生成a\n计算过程见下方展开图','state');down(uc,a)
    fc_op=box(600,665,'$F_c=a\\times U_c$\n每通道一个a，沿8×8位置广播','state');down(a,fc_op)
    s.text(ax,757,622,'原Uc',role='annotation')
    s.arrow(ax,[(775,641),(775,665),(fc_op['right'],665)])
    fs=box(200,765,'Fs [B,128,8,8]','output');down(prev,fs)
    fc=box(600,765,'Fc [B,128,8,8]','output');down(fc_op,fc)
    weights=box(400,900,'Fs、Fc共同生成两路融合权重\n$z=\\mathrm{Concat}(\\mathrm{GAP}(F_s),\\mathrm{GAP}(F_c))$\nMLP 256→32→256，再沿两路Softmax\n$a_s$、$a_c$初始各为0.5','output')
    # Each stream enters at its own top port. No opposing arrows or shared tips.
    for node,port in [(fs,weights['left']+30),(fc,weights['right']-30)]:
        s.arrow(ax,[(node['x'],node['bottom']),(node['x'],weights['top']-20),
                    (port,weights['top']-20),(port,weights['top'])])
    fuse=box(400,1080,'两路分别加权，再拼接并投影\n$F=\\mathrm{PW}_{256\\to256}(\\mathrm{Concat}(2a_sF_s,2a_cF_c))$','output');down(weights,fuse)
    post=box(400,1200,'LN → FFN(256→512→256) → GELU\nFFN内部含GELU；本处不额外加FFN输入','input');down(fuse,post)
    output=box(400,1310,'1×1投影：256→512通道','input');down(post,output)
    residual=box(400,1430,'与入口X相加\n$Y=X+\\gamma\\times output$\nY [B,512,8,8]；γ初值0.01','output');down(output,residual)
    s.text(ax,95,1409,'入口X',role='annotation')
    s.arrow(ax,[(95,1430),(residual['left'],1430)])
    s.compact_vertical(fig)
    s.save(fig,OUT/'SMB_详细')


def spectral():
    s=FigureStyle();fig,ax=s.canvas(1280);c=s.cfg['colors']
    def box(x,y,text,color):return s.box(ax,x,y,text,c[color])
    def down(a,b):s.arrow(ax,[(a['x'],a['bottom']),(b['x'],b['top'])])
    s.text(ax,400,30,'频谱通道加权：先统计，再把系数乘回原Uc',role='heading')
    uc=box(400,100,'Uc [B,128,8,8]','input')
    fft=box(290,235,'二维FFT后取功率\n$P=|\\mathrm{FFT2}(U_c)|^2$\nFP32计算，norm=ortho','state')
    kept=box(685,235,'保留原Uc\n空间特征不变','input')
    joint=uc['bottom']+28
    ax.plot([400,400],[uc['bottom'],joint],color=c['ink'],lw=1.25)
    ax.plot([290,685],[joint,joint],color=c['ink'],lw=1.25)
    for node in [fft,kept]:s.arrow(ax,[(node['x'],joint),(node['x'],node['top'])])
    energy=box(290,425,'每个通道分别统计低频／高频能量\n$E_L=\\sum_{(u,v)\\in L}P_{b,c,u,v}$\n$E_H=\\sum_{(u,v)\\notin L}P_{b,c,u,v}$','state');down(fft,energy)
    stats=box(290,650,'每通道得到3个统计数\n$\\log(1+E_L),\\quad\\log(1+E_H),\\quad r_H$\n$r_H=E_H/(E_L+E_H+10^{-8})$\n统计张量：[B,128,3]','state');down(energy,stats)
    mlp=box(290,850,'各通道共用同一个小网络\nMLP 3→16→1，再经2×Sigmoid\n输出a：[B,128]，初值1','state');down(stats,mlp)
    result=box(400,1095,'将a沿空间位置广播，再乘原Uc\n$F_c[b,c,h,w]=a[b,c]\\,U_c[b,c,h,w]$\n输出Fc：[B,128,8,8]','output')
    s.arrow(ax,[(290,mlp['bottom']),(290,result['top'])])
    port=result['right']-30
    s.arrow(ax,[(685,kept['bottom']),(685,result['top']-25),
                (port,result['top']-25),(port,result['top'])])
    s.text(ax,400,1220,'L：两个方向的频率绝对值均不超过0.25\nFFT结果用于生成a；Fc来自原空间Uc乘以a',role='annotation')
    s.compact_vertical(fig)
    s.save(fig,OUT/'SMB_频谱通道加权')


if __name__=='__main__':main();spectral()

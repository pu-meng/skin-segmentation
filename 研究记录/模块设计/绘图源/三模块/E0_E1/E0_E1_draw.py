"""Read-only code-faithful E0/E1 diagrams; isolated from experiment sources.
Run with medseg Python. Reuses existing Diagram and FigureStyle verbatim.
"""
from pathlib import Path
import sys,json,hashlib
ROOT=Path('/home/PuMengYu/experiments/skin_segmentation')
sys.path.insert(0,str(ROOT/'source/analysis/tinyvim_guide'))
from draw_block_details import Diagram
OUT=ROOT/'研究记录/模块设计/figures/三模块'
SOURCE=Path(__file__).resolve().parent

class D(Diagram):
    def __init__(self,title,height,subtitle=None):
        super().__init__(title,height)
        if subtitle:self.s.text(self.ax,400,76,subtitle,role='annotation')
    def box(self,key,x,y,text,color='projection',role=None):
        n=self.s.box(self.ax,x,y,text,self.s.cfg['colors'][color],role=role)
        self.nodes[key]=n
        return n
    def save(self,stem):
        audit=self.s.save(self.fig,OUT/stem)
        assert audit['passed'],audit['issues']
        print(stem,'saved',flush=True)
    def seq(self,rows,x=400,start=150,gap=140):
        keys=[]
        for i,(key,text,color) in enumerate(rows):
            self.box(key,x,start+i*gap,text,color);keys.append(key)
        self.chain(keys)


def overview(arm):
    wave=arm=='E1'
    title=f'{arm} · TinyViM-S 四层分割网络'
    d=D(title,1450, 'D-WPO ×1；完整替换 D3 ×1；其余原块保留' if wave else '新增模块 0 个；替换原块 0 个；四层 Decoder')
    d.box('in',132,180,'RGB 输入\nB×3×256×256','input',role='annotation')
    d.box('stem',132,290,'原 Stem · 保留\n两次步长2卷积\n3→24→48','projection',role='annotation')
    d.link('in','stem')
    ys=[440,715,990,1265];chs=[48,64,168,224];aligned=[64,128,256,512]
    for i,(y,c,a) in enumerate(zip(ys,chs,aligned)):
        size=64//2**i
        d.box(f's{i}',132,y,f'原 Stage {i} · 保留\n{c} ch · {size}×{size}','input',role='annotation')
        d.box(f'a{i}',365,y,f'原 1×1 对齐\n{c}→{a}\n'+(f'S{i} · {size}×{size}' if i<3 else '512 ch · 8×8'),'projection',role='annotation')
        d.link(f's{i}',f'a{i}',start='right',end='left')
        if i<3:
            d.box(f'down{i}',132,y+137,'原下采样 ×2\n卷积＋BN','projection',role='annotation')
            d.link(f's{i}',f'down{i}')
        if i>0:d.link(f'down{i-1}',f's{i}')
    d.link('stem','s0')
    for i,y in enumerate(ys):
        c=aligned[i];size=64//2**i
        if i==3:
            text=('D3：D-WPO ×1\n512→512 · 8×8\n完全替换 DoubleConv' if wave else 'D3：DoubleConv\n512→512 · 8×8\n无自拼接')
            color='state' if wave else 'output'
        else:
            text=f'拼接 S{i}：{c}＋{c}\nD{i}：DoubleConv\n{2*c}→{c} · {size}×{size}'
            color='output'
        d.box(f'd{i}',650,y,text,color,role='annotation')
        d.link(f'a{i}',f'd{i}',start='right',end='left')
        if i<3:
            d.box(f'up{i}',650,y+137,f'转置卷积 ×2\n{2*c}→{c} · k=2,s=2','projection',role='annotation')
            d.link(f'up{i}',f'd{i}',start='top',end='bottom')
    for i in range(3):d.link(f'd{i+1}',f'up{i}',start='top',end='bottom')
    d.box('out',650,190,'原输出头 · 保留\n1×1卷积：64→1 logits\n双线性插值至256×256\n再 Sigmoid → 前景概率','output',role='annotation')
    d.link('d0','out',start='top',end='bottom')
    d.s.text(d.ax,400,1390,'B：批量大小；ch：通道数。箭头传递特征；S0/S1/S2 是三条 skip。',role='annotation')
    d.save(f'{arm}_01_整网位置与模块数量')


def decoder():
    d=D('E0 · 四层 Decoder 怎样处理特征',1390,'D3 不拼接；只有 D2、D1、D0 接收三条 encoder skip')
    rows=[
    ('x','最深对齐特征\n[B,512,8,8]','input'),
    ('d3','D3 · DoubleConv\n512→512；8×8 不变','output'),
    ('d2','转置卷积 512→256，8→16\n拼接 S2：256＋256→512\nD2 · DoubleConv：512→256','projection'),
    ('d1','转置卷积 256→128，16→32\n拼接 S1：128＋128→256\nD1 · DoubleConv：256→128','projection'),
    ('d0','转置卷积 128→64，32→64\n拼接 S0：64＋64→128\nD0 · DoubleConv：128→64','projection'),
    ('head','1×1卷积 64→1\n[B,1,64,64] 原始 logits','output'),
    ('resize','双线性插值 logits\n64×64→256×256\nalign_corners=False','state'),
    ('prob','Sigmoid 逐元素映射至 (0,1)\n[B,1,256,256] 前景概率','output')]
    d.seq(rows,start=160,gap=160)
    d.save('E0_02_四层解码与输出顺序')
    d=D('E0 · DoubleConv 内部逐步计算',1010,'D3 / D2 / D1 / D0 共用同类结构；每个阶段各有独立参数')
    d.seq([
    ('x','阶段输入特征\n[B,Cin,H,W]','input'),
    ('c1','普通 3×3 卷积：Cin→Cout\nbias=False；步长1；padding=1','projection'),
    ('b1','BatchNorm → ReLU\n归一化后把负值置零','state'),
    ('c2','普通 3×3 卷积：Cout→Cout\nbias=False；步长1；padding=1','projection'),
    ('b2','BatchNorm → ReLU','state'),
    ('out','输出 [B,Cout,H,W]\n空间尺寸保持不变','output')],start=160,gap=150)
    d.save('E0_03_DoubleConv逐步计算')


def wave_flow():
    d=D('E1 · D-WPO 完整替换 D3 的内部计算',1500,'仅处理最深特征 [B,512,8,8]；不添加 E2/E3 的 pre / FFN / post')
    rows=[
    ('x','最深对齐特征 X\n[B,512,8,8]','input'),
    ('dw','保留官方 DW 3×3\n512通道；groups=512；bias=True','projection'),
    ('proj','逐位置 Linear：512→1536\n沿通道分成 U0、V0、Z，各512通道','projection'),
    ('fft','U0、V0 分别做正交 rFFT2\n得到复数频谱 F0、G0：8×5','state'),
    ('wave','方向性频域传播（见后两图）\n阻尼响应＋可学习权重的振荡响应\n产物：传播后的频谱 Fout','state'),
    ('ifft','正交 irFFT2：8×5→8×8\n每个位置沿512通道 LayerNorm','projection'),
    ('gate','逐元素乘 SiLU(Z)\n读取投影时保留的门控特征 Z','state'),
    ('outproj','保留官方输出 Linear：512→512\n得到 wave(X)','projection'),
    ('add','Y = X＋γ ⊙ wave(X)\nγ 为逐通道可学习系数，初始0.001','output'),
    ('to','Y 交给原转置卷积 512→256\n随后拼接 S2 并进入原 D2','output')]
    d.seq(rows,start=160,gap=135)
    d.bypass('x','add',735)
    # Z branches from the original shared projection, not from the FFT output.
    d.link('proj','gate',[(60,d.nodes['proj']['y']),(60,d.nodes['gate']['y'])],start='left',end='left')
    d.s.text(d.ax,72,740,'Z',role='annotation')
    d.save('E1_02_DWPO内部与残差')


def wave_parameters():
    d=D('E1 · 谁生成方向与传播系数',1350,'频率坐标由 8×8 网格确定；除时间 t 外，下列参数均可学习')
    rows=[
    ('freq','每个频率位置的坐标\nkx=2π·rfftfreq(8)，ky=2π·fftfreq(8)\nkx 对应水平方向，ky 对应垂直方向','input'),
    ('dir','逐通道方向参数 a，初始0\nax=1＋tanh(a)，ay=1−tanh(a)\nλ=ax·kx²＋ay·ky²','state'),
    ('p','正谱指数 p=softplus(p_raw＋p_ch)\np 初始为1；计算 λ 的 p 次幂\n实现用 exp(p·log(max(λ,1e−12)))','projection'),
    ('alpha','阻尼 α=max(softplus(α_grid＋α_ch),1e−4)\nα_grid 是 8×5×1 可学习参数\n网格初始 softplus 值0.05；再加通道偏置','projection'),
    ('boundary','rFFT 边界约束\n仅 kx=0 与 Nyquist 列\nα_grid 与反向 ky 行先取平均','state'),
    ('t','固定传播时间 t=1\nt_raw 与 t_ch 为固定 buffer，不训练\n头数1；每头512通道','input'),
    ('w','振荡混合参数 wave_weight\n每头一个可学习权重，初始0\n因此初始频域响应仅为阻尼项','output')]
    d.seq(rows,start=170,gap=170)
    # The boxes list parameter dependencies, not a sequential numerical pipeline.
    for patch in list(d.ax.patches):
        from matplotlib.patches import FancyArrowPatch
        if isinstance(patch,FancyArrowPatch):patch.remove()
    d.s.text(d.ax,400,1310,'各卡片说明不同系数的来源；不是依次修改同一张特征。',role='annotation')
    d.save('E1_03_方向与传播参数来源')


def wave_response():
    d=D('E1 · 频域响应怎样作用于 F0 和 G0',1210,'F0：初始场频谱；G0：初始速度频谱；λ、p、α、t 见参数图')
    d.box('in',400,170,'输入：F0、G0 与逐通道逐频率系数\nt=1；λ 体现水平／垂直方向权重','input')
    d.box('damp',205,400,'阻尼项 D\n'+r'$D=e^{-\alpha\lambda^p t}\,F_0$','state')
    d.box('om',590,380,'振荡频率\n'+r'$\omega=\sqrt{\lambda^p}$','projection')
    d.link('in','damp',[(400,270),(205,270)])
    d.link('in','om',[(400,270),(590,270)])
    d.box('wave',400,610,'振荡项 O：同时读取 F0 和 G0\n'+r'$O=e^{-\alpha\lambda^p t/2}\,[F_0\cos(\omega t)$'+'\n'+r'$+(G_0+\alpha\lambda^pF_0/2)\sin(\omega t)/\omega]$','state')
    d.link('om','wave',[(590,490),(400,490)])
    d.link('in','wave',[(742,170),(742,610)],start='right',end='right')
    d.s.text(d.ax,680,500,'F0 / G0',role='annotation')
    d.box('mix',400,840,'按头混合两种响应\n'+r'$F_{out}=D+w\,(O-D)$'+'\nw = wave_weight，初始0','projection')
    d.link('wave','mix')
    d.link('damp','mix',[(48,400),(48,840)],start='left',end='left')
    d.box('out',400,1050,'输出复数频谱 Fout\n再 irFFT2 → LayerNorm → SiLU(Z)门控\n→ 输出 Linear → 乘γ并加回 X','output')
    d.link('mix','out')
    d.s.text(d.ax,400,1160,'FFT、幂指数与复数响应使用 FP32；ω 分母按实现夹到至少 1e−6。',role='annotation')
    d.save('E1_04_阻尼与振荡响应逐步计算')

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--batch',choices=['overview','details','all'],default='all');args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.batch in ('overview','all'):
        overview('E0');overview('E1')
    if args.batch in ('details','all'):
        decoder();wave_flow();wave_parameters();wave_response()

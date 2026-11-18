"""Operator-level diagrams for the three blocks, from active local code.
All fonts, fitted nodes and exports are provided by the shared renderer.
"""
from pathlib import Path
import sys
sys.path.insert(0, '/home/PuMengYu/.codex/skills/research-figure-style/scripts')
from figure_style import FigureStyle
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / '研究记录/figures'


class Diagram:
    def __init__(self, title, height):
        self.s = FigureStyle()
        self.fig, self.ax = self.s.canvas(height)
        self.nodes = {}
        self.s.text(self.ax, 400, 28, title, role='heading')

    def box(self, key, x, y, text, color='projection'):
        n = self.s.box(self.ax, x, y, text, self.s.cfg['colors'][color])
        self.nodes[key] = n
        return n

    def link(self, a, b, via=None, start='bottom', end='top'):
        a, b = self.nodes[a], self.nodes[b]
        p = (a['x'], a[start]) if start in ('top','bottom') else (a[start], a['y'])
        q = (b['x'], b[end]) if end in ('top','bottom') else (b[end], b['y'])
        self.s.arrow(self.ax, [p] + (via or []) + [q])

    def chain(self, keys):
        for a, b in zip(keys, keys[1:]):
            self.link(a, b)

    def bypass(self, a, b, x, side='right'):
        a0, b0 = self.nodes[a], self.nodes[b]
        self.link(a, b, [(x, a0['y']), (x, b0['y'])], start=side, end=side)

    def save(self, stem):
        self.s.compact_vertical(self.fig)
        audit = self.s.save(self.fig, OUT / stem)
        print(stem, len(audit['boxes']), 'boxes; passed', flush=True)


def local():
    d = Diagram('LocalBlock：RepDW → FFN → 缩放 → 残差', 1440)
    d.box('x',400,105,'输入 X\n[B,C,H,W]','input')
    d.box('dw',175,255,'DWConv 3×3\ngroups=C · bias=False\n步长1 · padding=1')
    d.box('bn0',175,370,'BN')
    d.box('pw',435,255,'DWConv 1×1\ngroups=C · bias=True\n步长1 · padding=0')
    d.box('id',655,255,'恒等\nX','input')
    for k in ('dw','pw','id'):
        d.link('x',k,[(400,180),(d.nodes[k]['x'],180)])
    d.link('dw','bn0')
    d.box('sum',400,475,'＋','state')
    for k in ('bn0','pw','id'):
        d.link(k,'sum',[(d.nodes[k]['x'],425),(400,425)])
    d.box('bn1',400,555,'BN')
    d.box('fc1',400,640,'1×1卷积\nC→4C · bias=False')
    d.box('bn2',400,735,'BN → GELU\n批归一化 → 平滑激活','state')
    d.box('fc2',400,830,'1×1卷积\n4C→C · bias=False')
    d.box('bn3',400,915,'BN')
    d.box('drop',400,990,'Dropout · p=0','state')
    d.box('scale',400,1095,'LayerScale\n'+r'$\gamma\odot x$'+'\nγ：C×1×1，初始为1','state')
    d.box('path',400,1210,'DropPath · p=0','state')
    d.box('add',400,1290,'＋ X','state')
    d.box('out',400,1380,'输出\n[B,C,H,W]','output')
    d.chain(['sum','bn1','fc1','bn2','fc2','bn3','drop','scale','path','add','out'])
    d.bypass('x','add',765)
    d.s.text(d.ax,95,530,'RepDW',role='annotation')
    d.s.text(d.ax,170,820,'FFN',role='heading')
    d.save('25_LocalBlock计算')


def tvim():
    d = Diagram('TViMBlock：Mixer 残差 → FFN 残差', 1220)
    specs = [
        ('x',100,'输入 X\n[B,C,H,W]','input'),
        ('mix',225,'Laplace Mixer\nSS2D · 内部见下图','projection'),
        ('p1',325,'DropPath · p=0','state'),
        ('a1',410,'＋ X','state'),
        ('fc1',510,'1×1卷积\nC→4C · bias=False','projection'),
        ('bn1',610,'BN → GELU\n批归一化 → 平滑激活','state'),
        ('fc2',710,'1×1卷积\n4C→C · bias=False','projection'),
        ('bn2',800,'BN','projection'),
        ('drop',875,'Dropout · p=0','state'),
        ('p2',950,'DropPath · p=0','state'),
        ('a2',1035,'＋','state'),
        ('out',1140,'输出\n[B,C,H,W]','output'),
    ]
    for key,y,text,color in specs:
        d.box(key,400,y,text,color)
    d.chain([x[0] for x in specs])
    d.bypass('x','a1',710)
    d.bypass('a1','a2',120,side='left')
    d.s.text(d.ax,200,695,'FFN',role='heading')
    d.save('28_TViMBlock计算')


def mixer():
    d = Diagram('Laplace Mixer（SS2D）：以 Stage 2 为例', 1460)
    d.box('x',400,105,'输入\n[B,168,16,16]','input')
    d.box('proj',400,220,'1×1卷积 → BN\n168→168 · bias=False')
    d.box('axis',400,325,'轴向 Rep_Inception\n内部见下一图')
    d.box('silu',400,415,'SiLU','state')
    d.box('split',400,500,'按通道顺序拆分\n前84通道 → 低路\n后84通道 → 高路','input')
    d.chain(['x','proj','axis','silu','split'])
    d.box('pool',200,665,'平均池化 2×2\n步长2；16×16→8×8','state')
    d.box('near',440,780,'最近邻上采样\n8×8→16×16\nu：池化结果放大回原尺寸','state')
    d.box('res',440,930,'细节相减\n'+r'$res=X_{low}-u$','state')
    d.box('scan',200,900,'四方向扫描\nCrossScan → SSM\nCrossMerge → LN\n84 ch · 8×8','state')
    d.box('up',200,1050,'双线性上采样\n8×8→16×16','state')
    d.box('add',200,1150,'＋ res','state')
    d.box('high',680,665,'RepDW 3×3\n接后84通道\n84 ch · 16×16')
    d.link('split','pool',[(400,565),(200,565)])
    d.link('split','high',[(400,550),(680,550)])
    # Original low feature forks before pooling. The detail branch is a subtraction.
    d.s.arrow(d.ax,[(200,565),(575,565),(575,930),(d.nodes['res']['right'],930)])
    d.s.text(d.ax,460,598,r'$X_{low}$',role='annotation')
    d.link('pool','near',[(440,665)],start='right')
    d.link('near','res')
    d.link('pool','scan')
    d.link('scan','up')
    d.link('up','add')
    d.link('res','add',[(440,1150)],end='right')
    d.box('cat',400,1250,'通道拼接\n84＋84 → 168','projection')
    d.link('add','cat',[(200,1250)],end='left')
    d.link('high','cat',[(680,1250)],end='right')
    d.box('out',400,1360,'1×1卷积 → BN\n168→168 · bias=False\n[B,168,16,16]','output')
    d.link('cat','out')
    d.s.text(d.ax,400,1430,'Stage 0/1：池化8/4倍；Stage 3：无池化、无 res。',role='annotation')
    d.save('29_LaplaceMixer计算')


def axial():
    d = Diagram('Rep_Inception：两组通道，各走三条分支', 960)
    d.box('x',400,105,'按通道对半分组\n168 → 84＋84','input')
    d.box('g1',220,220,'前84通道','input')
    d.box('g2',590,220,'后84通道','input')
    d.link('x','g1',[(400,170),(220,170)])
    d.link('x','g2',[(400,170),(590,170)])
    d.box('dw1',130,370,'DW 7×1\npadding=(3,0)\nbias=False')
    d.box('pw1',305,370,'DW 1×1\nbias=True')
    d.box('dw2',495,370,'DW 1×7\npadding=(0,3)\nbias=False')
    d.box('pw2',675,370,'DW 1×1\nbias=True')
    for key in ('dw1','pw1'):
        d.link('g1',key,[(220,280),(d.nodes[key]['x'],280)])
    for key in ('dw2','pw2'):
        d.link('g2',key,[(590,280),(d.nodes[key]['x'],280)])
    for i,x in [(1,220),(2,590)]:
        d.box(f'bn{i}',d.nodes[f'dw{i}']['x'],490,'BN')
        d.box(f'sum{i}',x,590,'＋','state')
        d.box(f'out{i}',x,690,'BN\n84 ch · 16×16')
        d.link(f'dw{i}',f'bn{i}')
        for key in (f'bn{i}',f'pw{i}'):
            d.link(key,f'sum{i}',[(d.nodes[key]['x'],540),(x,540)])
        d.link(f'sum{i}',f'out{i}')
    # Each channel group retains its own identity branch.
    d.s.arrow(d.ax,[(220,280),(25,280),(25,590),(d.nodes['sum1']['left'],590)])
    d.s.arrow(d.ax,[(590,280),(780,280),(780,590),(d.nodes['sum2']['right'],590)])
    d.s.text(d.ax,65,530,'恒等',role='annotation')
    d.s.text(d.ax,740,530,'恒等',role='annotation')
    d.box('cat',400,825,'通道拼接\n[B,168,16,16]','output')
    d.link('out1','cat',[(220,755),(400,755)])
    d.link('out2','cat',[(590,755),(400,755)])
    d.s.text(d.ax,400,920,'DW：深度卷积，groups=84，步长1；1×1卷积 padding=0。',role='annotation')
    d.save('30_轴向卷积RepInception')


def doubleconv():
    d = Diagram('DoubleConv：两次 Conv → BN → ReLU', 915)
    d.box('x',400,110,'输入：拼接后的特征\n512 ch · 16×16','input')
    d.box('c1',400,270,'3×3卷积\n512→256 · bias=False\n步长1 · padding=1')
    d.box('b1',400,420,'BN → ReLU\n批归一化 → 负值置零','state')
    d.box('c2',400,570,'3×3卷积\n256→256 · bias=False\n步长1 · padding=1')
    d.box('b2',400,720,'BN → ReLU\n批归一化 → 负值置零','state')
    d.box('out',400,850,'输出\n256 ch · 16×16','output')
    d.chain(['x','c1','b1','c2','b2','out'])
    d.save('33_Decoder拼接与DoubleConv')


def directions():
    d = Diagram('四方向扫描：同一特征，四种顺序', 1060)
    d.box('x',400,110,'同一低路特征\n示例：第一行1 2 3，第二行4 5 6\n[B,84,8,8]','input')
    orders = [('行正向','1 2 3 4 5 6'), ('列正向','1 4 2 5 3 6'),
              ('行反向','6 5 4 3 2 1'), ('列反向','6 3 5 2 4 1')]
    for i, (x, (title, seq)) in enumerate(zip((110,305,500,695), orders)):
        d.box(f'order{i}',x,320,f'{title}\n{seq}\n84 ch · 64位置')
        d.box(f'scan{i}',x,495,'独立 SSM\n生成 Δ、B、C','state')
        d.link('x',f'order{i}',[(400,220),(x,220)])
        d.link(f'order{i}',f'scan{i}')
    d.box('merge',400,690,'撤销反向／转置\n对齐空间位置 → 四路求和\n[B,84,64]','projection')
    for i in range(4):
        x = d.nodes[f'scan{i}']['x']
        d.link(f'scan{i}','merge',[(x,595),(400,595)])
    d.box('norm',400,850,'转置 → LayerNorm\n[B,64,84]；沿84通道归一化','state')
    d.box('out',400,990,'恢复二维布局\n[B,84,8,8]','output')
    d.chain(['merge','norm','out'])
    d.save('31_四方向扫描与还原')


def draw_all():
    local()
    tvim()
    mixer()
    axial()
    directions()
    doubleconv()
    from draw_tvim_pair import draw as draw_pair
    draw_pair()


if __name__ == '__main__':
    draw_all()

"""Native two-panel layout at the shared 800px width; no image downscaling.
Left: TViMBlock. Right: its Mixer. Operators match draw_block_details.py.
"""
from draw_block_details import Diagram


def draw():
    d = Diagram('TViMBlock 与 Mixer：左右对照', 1590)
    d.s.text(d.ax,150,75,'图3 · TViMBlock',role='heading')
    d.s.text(d.ax,555,75,'图4 · Laplace Mixer',role='heading')
    left_specs = [
        ('lx',160,'输入 X\n[B,C,H,W]','input'),
        ('lm',285,'Laplace Mixer\nSS2D · 见右图4','projection'),
        ('lp1',385,'DropPath · p=0','state'),
        ('la1',470,'＋ X','state'),
        ('lf1',570,'1×1卷积\nC→4C · bias=False','projection'),
        ('lb1',670,'BN → GELU\n批归一化 → 平滑激活','state'),
        ('lf2',770,'1×1卷积\n4C→C · bias=False','projection'),
        ('lb2',860,'BN','projection'),
        ('ld',935,'Dropout · p=0','state'),
        ('lp2',1010,'DropPath · p=0','state'),
        ('la2',1095,'＋','state'),
        ('lo',1200,'输出\n[B,C,H,W]','output'),
    ]
    for key,y,text,color in left_specs:
        d.box(key,150,y,text,color)
    d.chain([row[0] for row in left_specs])
    d.bypass('lx','la1',285)
    d.bypass('la1','la2',18,side='left')
    d.s.text(d.ax,150,1310,'FFN：两次1×1卷积\nC → 4C → C',role='annotation')

    # The right panel preserves all split, subtraction and recovery branches.
    d.box('x',555,160,'输入 · Stage 2\n[B,168,16,16]','input')
    d.box('proj',555,270,'1×1卷积 → BN\n168→168 · bias=False')
    d.box('axis',555,380,'轴向卷积\nRep_Inception · 见图5')
    d.box('silu',555,465,'SiLU','state')
    d.box('split',555,555,'按通道顺序拆分\n前84通道 → 低路\n后84通道 → 高路','input')
    d.chain(['x','proj','axis','silu','split'])
    d.box('pool',410,735,'平均池化2×2\n16×16→8×8','state')
    d.box('near',610,865,'最近邻上采样\nu：池化结果\n放大回16×16','state')
    d.box('res',610,1030,'细节相减\n'+r'$res=X_{low}-u$','state')
    d.box('scan',410,1000,'四方向扫描\nCrossScan → SSM\nCrossMerge → LN\n84 ch · 8×8','state')
    d.box('up',410,1150,'双线性上采样\n8×8→16×16','state')
    d.box('add',410,1250,'＋ res','state')
    d.box('high',700,1250,'RepDW 3×3\n后84通道\n16×16')
    d.link('split','pool',[(555,640),(410,640)])
    d.link('split','high',[(555,620),(780,620),(780,1140),(700,1140)])
    d.s.arrow(d.ax,[(410,640),(725,640),(725,1030),(d.nodes['res']['right'],1030)])
    d.s.text(d.ax,665,675,r'$X_{low}$',role='annotation')
    d.link('pool','near',[(610,735)],start='right')
    d.chain(['near','res'])
    d.chain(['pool','scan','up','add'])
    d.link('res','add',[(610,1250)],end='right')
    d.box('cat',555,1350,'通道拼接\n84＋84 → 168')
    d.link('add','cat',[(410,1350)],end='left')
    d.link('high','cat',[(700,1350)],end='right')
    d.box('out',555,1470,'1×1卷积 → BN\n168→168 · bias=False\n[B,168,16,16]','output')
    d.link('cat','out')
    d.s.text(d.ax,555,1560,'Stage 0/1：池化8/4倍\nStage 3：无池化、无 res',role='annotation')
    d.save('35_TViMBlock与Mixer左右对照')


if __name__ == '__main__':
    draw()

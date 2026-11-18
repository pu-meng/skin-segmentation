"""Code-faithful shallow refiners and parameter-matched bottleneck controls.

Sources: Models/ShallowSkipRefiners/model.py and
Models/TinyViM/bottleneck_parameter_controls.py. Style remains shared.
"""
from pathlib import Path
import sys

sys.path.insert(0, '/home/PuMengYu/.codex/skills/research-figure-style/scripts')
from figure_style import FigureStyle

OUT = Path(__file__).resolve().parents[3] / '研究记录/figures'


def stack(stem, title, specs, bypasses=()):
    s = FigureStyle()
    fig, ax = s.canvas()
    s.text(ax, 400, 30, title, role='heading')
    nodes = []
    top = 78
    for text, color in specs:
        width, height = s.measure(ax, text)
        if width > 650:
            raise ValueError(f'请拆行：{text}')
        node = s.box(ax, 400, top+height/2, text, s.cfg['colors'][color])
        if nodes:
            s.arrow(ax, [(400, nodes[-1]['bottom']), (400, node['top'])])
        nodes.append(node)
        top = node['bottom']+42
    # Original input / membership dependencies must reach the actual consumer.
    for source, target, side, label in bypasses:
        a, b = nodes[source], nodes[target]
        route = 38 if side == 'left' else 762
        s.arrow(ax, [(a[side], a['y']), (route, a['y']),
                     (route, b['y']), (b[side], b['y'])])
        s.text(ax, (route+b[side])/2, b['y']-18, label, role='annotation')
    height = nodes[-1]['bottom']+30
    fig.set_size_inches(s.cfg['output']['width_px']/s.cfg['output']['dpi'], height/s.cfg['output']['dpi'])
    ax.set_ylim(height, 0)
    audit = s.save(fig, OUT/stem)
    if not audit['passed']:
        raise ValueError(audit['issues'])
    print(stem, flush=True)


def draw_all():
    stack('50_V0最浅Skip空间门控', 'V0＋最浅Skip空间门控', [
        ('1. 取原有两路特征\nS0：Stage 0经1×1对齐后的skip\nU2：D1输出经D2转置卷积后的特征\n两路均为 [B,64,64,64]', 'input'),
        ('2. 两个独立的无偏置1×1卷积\nS0与U2各自64→32\n两路均为 [B,32,64,64]', 'projection'),
        ('3. 两路逐元素相加，再ReLU\n[B,32,64,64]', 'state'),
        ('4. 生成每个位置的一个权重\n1×1卷积32→1（有偏置）→ Sigmoid\n空间权重：[B,1,64,64]', 'state'),
        ('5. 原S0乘空间权重\n沿64个通道广播同一个位置权重\n输出：[B,64,64,64]', 'state'),
        ('6. 送回V0原有D2\n与U2拼接：[B,128,64,64]\n原DoubleConv：128→64→64', 'output'),
    ], [(0, 4, 'left', '原S0'), (0, 5, 'right', '原U2')])
    stack('51_V0最浅Skip通道门控', 'V0＋最浅Skip通道门控', [
        ('1. 取对齐后的最浅Skip\nS0：[B,64,64,64]', 'input'),
        ('2. 沿高、宽求平均\n64×64位置 → 每通道一个数\n[B,64,1,1]', 'projection'),
        ('3. 1×1卷积64→8，再ReLU\n卷积有偏置；[B,8,1,1]', 'state'),
        ('4. 1×1卷积8→64，再Sigmoid\n卷积有偏置；通道权重 [B,64,1,1]', 'state'),
        ('5. 原S0乘通道权重\n沿64×64位置广播每通道的权重\n输出：[B,64,64,64]', 'state'),
        ('6. 送回V0原有D2\n与U2拼接：[B,128,64,64]\n原DoubleConv：128→64→64', 'output'),
    ], [(0, 4, 'left', '原S0')])
    stack('52_V0最浅Skip区域重构', 'V0＋最浅Skip区域重构', [
        ('1. 把S0的空间位置排成序列\n[B,64,64,64] → T：[B,4096,64]\n每个位置保留64维原始特征', 'input'),
        ('2. 与8个可学习中心比较\n中心：[8,64]；位置与中心各沿通道L2归一化\n余弦相似度除以0.1\n分数：[B,4096,8]', 'projection'),
        ('3. 每个位置分配给8个区域\n沿8个区域Softmax，得到M\nM：[B,4096,8]；每个位置权重和为1', 'state'),
        ('4. 为每张图汇总8份区域特征\nM沿4096个位置再归一化，得到W\n'+r'$R=W^\top T$'+'\nR：[B,8,64]；使用原始T汇总', 'state'),
        ('5. 区域特征按原分配权重送回位置\n'+r"$T'=MR$"+'\n[B,4096,8] × [B,8,64] → [B,4096,64]', 'state'),
        ('6. 还原并送回V0原有D2\n[B,64,64,64]，直接替代S0\n与U2拼接 → 原DoubleConv', 'output'),
    ], [(0, 3, 'left', '原T'), (2, 4, 'right', '原M')])
    for kind, width, stem, name in [
        ('conv', 184, '53_V0等参数瓶颈普通卷积', '普通卷积'),
        ('mlp', 1653, '54_V0等参数瓶颈MLP', 'MLP'),
    ]:
        conv = kind == 'conv'
        stack(stem, f'V0＋等参数瓶颈{name}', [
            ('1. 取最深层原有对齐输出X\nStage 3：224通道，经1×1卷积对齐\nX：[B,512,8,8]', 'input'),
            ('2. 展平位置，逐位置LayerNorm\n[B,512,8,8] → [B,64,512]\n沿512通道归一化；有可学习缩放与偏置', 'projection'),
            (('3. 还原空间，再做普通3×3卷积\n512→184，padding1，groups1\n[B,184,8,8]' if conv else
              '3. 每个位置做Linear\n512→1653；有偏置\n[B,64,1653]'), 'state'),
            (f'4. GELU激活\n'+('[B,184,8,8]' if conv else '[B,64,1653]'), 'state'),
            (('5. 第二个普通3×3卷积\n184→512，padding1，groups1\n两层卷积都有偏置；[B,512,8,8]' if conv else
              '5. 第二个Linear\n1653→512；有偏置\n[B,64,512]'), 'state'),
            (('6. 与原输入X逐元素相加\n输出：[B,512,8,8]\n送入D0第一次上采样，原有Skip保留' if conv else
              '6. 加回原X展平后的序列，再还原\n[B,64,512] → [B,512,8,8]\n送入D0第一次上采样，原有Skip保留'), 'output'),
        ], [(0, 5, 'left', '原X')])


if __name__ == '__main__':
    draw_all()

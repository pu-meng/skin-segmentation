"""Trace ConvTransform and decode_features; reuse the shared FigureStyle."""
from pathlib import Path
import sys

sys.path.insert(0, '/home/PuMengYu/.codex/skills/research-figure-style/scripts')
from figure_style import FigureStyle

OUT = Path(__file__).resolve().parents[3] / '研究记录/figures'


def draw():
    s = FigureStyle()
    fig, ax = s.canvas()
    colors = s.cfg['colors']
    s.text(ax, 400, 30, 'V0＋最浅Skip卷积变换', role='heading')
    specs = [
        ('1. Stage 0原始特征\n[B,48,64,64]', 'input'),
        ('2. 原有1×1通道对齐\n48→64；输出S0\n[B,64,64,64]', 'projection'),
        ('3. 新增逐通道3×3卷积\n64个通道各用自己的3×3核\n步长1，padding1，groups64\n[B,64,64,64]', 'state'),
        ('4. 新增1×1通道混合\n同一位置的64通道组合成64通道\n[B,64,64,64]', 'state'),
    ]
    top = 75
    nodes = []
    for text, color in specs:
        _, height = s.measure(ax, text)
        node = s.box(ax, 500, top + height/2, text, colors[color])
        if nodes:
            s.arrow(ax, [(500, nodes[-1]['bottom']), (500, node['top'])])
        nodes.append(node)
        top = node['bottom'] + 42
    context_text = 'D1的输出\n[B,128,32,32]'
    _, ch = s.measure(ax, context_text)
    context = s.box(ax, 145, nodes[0]['top']+ch/2, context_text, colors['input'])
    up_text = 'D2原有转置卷积\n128→64，上采样2倍\nU2：[B,64,64,64]'
    _, uh = s.measure(ax, up_text)
    up = s.box(ax, 145, nodes[-1]['bottom']-uh/2, up_text, colors['projection'])
    s.arrow(ax, [(145, context['bottom']), (145, up['top'])])
    text = '5. 两路沿通道拼接\nU2与变换后的S0 → [B,128,64,64]'
    _, height = s.measure(ax, text)
    concat = s.box(ax, 400, top+height/2, text, colors['output'])
    s.arrow(ax, [(up['x'], up['bottom']), (up['x'], concat['y']),
                 (concat['left'], concat['y'])])
    s.arrow(ax, [(500, nodes[-1]['bottom']), (500, concat['top'])])
    top = concat['bottom']+42
    text = '6. D2原有双卷积\n128→64→64；各接BN与ReLU\n[B,64,64,64] → 继续D3、D4'
    _, height = s.measure(ax, text)
    output = s.box(ax, 400, top+height/2, text, colors['output'])
    s.arrow(ax, [(400, concat['bottom']), (400, output['top'])])
    height = output['bottom']+55
    s.text(ax, 400, height-22, '黄色为新增两层；Stage 0向深层编码的连接保持原计算。', role='annotation')
    fig.set_size_inches(s.cfg['output']['width_px']/s.cfg['output']['dpi'], height/s.cfg['output']['dpi'])
    ax.set_ylim(height, 0)
    audit = s.save(fig, OUT/'49_基础模型最浅Skip卷积变换')
    if not audit['passed']:
        raise ValueError(audit['issues'])


if __name__ == '__main__':
    draw()

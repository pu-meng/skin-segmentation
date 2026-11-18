"""Bottleneck frequency branches, traced from mamba_bottleneck_v2_frequency.py.

All typography, measured boxes and previews use the shared FigureStyle.
"""
from pathlib import Path
import sys

sys.path.insert(0, '/home/PuMengYu/.codex/skills/research-figure-style/scripts')
from figure_style import FigureStyle

OUT = Path(__file__).resolve().parents[3] / '研究记录/figures'


def draw(dynamic=False):
    s = FigureStyle()
    width = s.cfg['output']['width_px']
    fig, ax = s.canvas()
    colors = s.cfg['colors']
    title = 'V2＋动态低秩频域' if dynamic else 'V2＋固定低秩频域'
    generator = (
        '2. 每张图生成低秩因子\n空间均值：[B,512] → Linear → GELU\n'
        '隐藏维度32，再用两个Linear生成\nU：[B,512,4]；V：[B,4,40]\n'
        r'$P_b=U_bV_b$'
        if dynamic else
        '2. 全部图片共用可学习因子\nU：[512,4]；V：[4,40]\n'
        r'$P=UV$' + '\nP：[512,40]；沿batch广播'
    )
    specs = [
        ('input', '1. 最深层对齐后的特征\nX：[B,512,8,8]', 'input'),
        ('generator', generator, 'projection'),
        ('gain', '3. 生成实数频率增益\n' + r'$G=1+0.1\tanh(P)$'
         + '\n恢复 [B,512,8,5]\n边界列沿高度配对平均', 'state'),
        ('fft', '4. 对原输入做二维FFT，再乘增益\n'
         + r'$G\odot\mathrm{rFFT2}(X)$'
         + '\n每通道独立；频谱 [B,512,8,5]', 'projection'),
        ('inverse', '5. 逆FFT恢复空间特征\n'
         + r'$\mathrm{irFFT2}(G\odot\mathrm{rFFT2}(X))$'
         + '\n[B,512,8,8]；FFT支路使用FP32', 'projection'),
        ('output', '6. 频域结果乘系数，再加到原V2输出\n'
         + r'$Y=\mathrm{V2}(X)+\alpha\,\mathrm{irFFT2}(G\odot\mathrm{rFFT2}(X))$'
         + '\nα可学习，初值0.01；输出 [B,512,8,8]\n送入decoder第一次上采样', 'output'),
    ]
    sizes = {key: s.measure(ax, text) for key, text, _ in specs}
    base_text = '原V2支路\n' + r'$\mathrm{V2}(X)$' + '\n内部计算保留'
    bw, bh = s.measure(ax, base_text)
    freq_w = max(sizes[k][0] for k in ('generator', 'gain', 'fft', 'inverse'))
    gap = s.cfg['box']['node_gap']
    total = bw + gap + freq_w
    if total > width - 90:
        raise ValueError('Frequency branches exceed shared canvas width')
    left = (width - total) / 2
    base_x = left + bw / 2
    freq_x = left + bw + gap + freq_w / 2
    nodes = {}
    top = 65
    for key, text, color in specs:
        w, h = sizes[key]
        x = width / 2 if key in ('input', 'output') else freq_x
        nodes[key] = s.box(ax, x, top + h / 2, text, colors[color])
        top += h + gap
    start, end = nodes['generator'], nodes['inverse']
    base_y = (start['top'] + end['bottom']) / 2
    nodes['base'] = s.box(ax, base_x, base_y, base_text, colors['state'])
    height = top + 15
    fig.set_size_inches(width / s.cfg['output']['dpi'], height / s.cfg['output']['dpi'])
    ax.set_ylim(height, 0)
    s.text(ax, width / 2, 27, title + ' · 512通道、8×8瓶颈', role='heading')
    inp, out, base = nodes['input'], nodes['output'], nodes['base']
    fork_y = (inp['bottom'] + start['top']) / 2
    for n in (start, base):
        s.arrow(ax, [(inp['x'], inp['bottom']), (inp['x'], fork_y),
                     (n['x'], fork_y), (n['x'], n['top'])])
    for a, b in [('generator', 'gain'), ('gain', 'fft'), ('fft', 'inverse')]:
        s.arrow(ax, [(nodes[a]['x'], nodes[a]['bottom']), (nodes[b]['x'], nodes[b]['top'])])
    join_y = (end['bottom'] + out['top']) / 2
    for n in (base, end):
        s.arrow(ax, [(n['x'], n['bottom']), (n['x'], join_y),
                     (out['x'], join_y), (out['x'], out['top'])])
    # FFT consumes the original X independently of how G was produced.
    route = max(nodes[k]['right'] for k in ('generator', 'gain', 'fft', 'inverse')) + 18
    fft = nodes['fft']
    if route > width - 8:
        raise ValueError('Input-to-FFT route exceeds canvas')
    s.arrow(ax, [(inp['right'], inp['y']), (route, inp['y']),
                 (route, fft['y']), (fft['right'], fft['y'])])
    s.text(ax, route - 12, fft['y'] - 24, '$X$', role='annotation')
    stem = '39_V2动态低秩频域' if dynamic else '38_V2固定低秩频域'
    audit = s.save(fig, OUT / stem)
    if not audit['passed']:
        raise ValueError(audit['issues'])
    print(stem, flush=True)


def draw_all():
    draw(False)
    draw(True)


if __name__ == '__main__':
    draw_all()

"""V4 depthwise-separable branches, elementwise sum and baseline D0 routing."""
from draw_baseline import FigureStyle, ROOT


def draw():
    s = FigureStyle()
    fig, ax = s.canvas(1570)
    c = s.cfg['colors']
    nodes = {}
    enc_x, skip_x, dec_x = 155, 380, 665

    def box(key, x, y, text, color):
        nodes[key] = s.box(ax, x, y, text, c[color])
        return nodes[key]

    def link(a, b):
        a, b = nodes[a], nodes[b]
        if a['x'] == b['x']:
            start, end = ('bottom', 'top') if a['y'] < b['y'] else ('top', 'bottom')
            points = [(a['x'], a[start]), (b['x'], b[end])]
        else:
            points = [(a['right'], a['y']), (b['left'], b['y'])]
        s.arrow(ax, points)

    s.text(ax, 400, 28, 'V4：基线上的多层Skip汇聚到D0', role='heading')
    s.text(ax, enc_x, 73, 'TinyViM-S encoder', role='annotation')
    s.text(ax, dec_x, 73, 'UNet decoder · D0→D4', role='annotation')
    box('input', enc_x, 140, 'RGB输入与归一化\n3 ch · 256×256', 'input')
    box('stem', enc_x, 310, 'Stem\n两次步长2卷积\n48 ch · 64×64', 'projection')
    link('input', 'stem')
    ys = (500, 740, 980, 1430)
    channels = (48, 64, 168, 224)
    aligned = (64, 128, 256, 512)
    seqs = ('Local×2 → TViM', 'Local×2 → TViM',
            'Local×4 → TViM\nLocal×3 → TViM', 'Local×5 → TViM\n末尾BN')
    for i, y in enumerate(ys):
        size = 64 // 2**i
        box(f'e{i}', enc_x, y, f'Encoder Stage {i}\n{seqs[i]}\n{channels[i]} ch · {size}×{size}', 'input')
        prefix = f'S{i} · 1×1投影' if i < 3 else '最深层 · 1×1投影'
        box(f's{i}', skip_x, y, f'{prefix}\n{channels[i]}→{aligned[i]}\n{aligned[i]} ch · {size}×{size}', 'projection')
        link(f'e{i}', f's{i}')
        if i < 3:
            down_y = (y + ys[i+1]) / 2
            box(f'down{i}', enc_x, down_y, f'下采样 ×2\n{channels[i]}→{channels[i+1]}', 'projection')
            link(f'e{i}', f'down{i}')
    link('stem', 'e0')
    for i in range(3):
        link(f'down{i}', f'e{i+1}')

    box('gather', 435, 1220, 'DW3×3＋PW1×1\n各256 ch · 16×16\n三路逐元素相加', 'state')
    gather = nodes['gather']
    # Higher skips use the outer lanes, so lower horizontal branches do not cross.
    for i, route in enumerate((520, 505, 490)):
        node = nodes[f's{i}']
        assert route > node['right'] and gather['left'] < route < gather['right']
        s.arrow(ax, [(node['right'], node['y']), (route, node['y']), (route, gather['top'])])
    box('d0', dec_x, 1220, 'D0 · 最深层\n拼接→DoubleConv\n256 ch · 16×16', 'output')
    link('gather', 'd0')
    box('up0', dec_x, 1430, '上采样 ×2\n512→256\n8×8 → 16×16', 'state')
    link('s3', 'up0'); link('up0', 'd0')
    for i, y, ch, size in [(1, 980, 128, 32), (2, 740, 64, 64),
                           (3, 500, 32, 128), (4, 280, 16, 256)]:
        extra = f'\n保留原S{2-i}连接' if i in (1, 2) else ''
        box(f'd{i}', dec_x, y, f'D{i} · DoubleConv\n{ch} ch · {size}×{size}{extra}', 'output')
        prev = nodes[f'd{i-1}']
        up_y = (prev['y'] + y) / 2
        box(f'up{i}', dec_x, up_y, f'上采样 ×2\n{ch*2}→{ch}', 'state')
        link(f'd{i-1}', f'up{i}'); link(f'up{i}', f'd{i}')
    box('output', dec_x, 140, '1×1卷积＋Sigmoid\n1 ch · 256×256', 'output')
    link('d4', 'output')
    s.text(ax, 400, 1523, '原始基线＋多层Skip融合；DW步长4／2／1；S0→D2、S1→D1保留。', role='annotation')
    s.compact_vertical(fig)
    audit = s.save(fig, ROOT/'研究记录/figures/40_V4多层Skip汇聚到D0总图')
    if not audit['passed']:
        raise ValueError(audit['issues'])
    print('V4 overview: shared-style audit passed.', flush=True)


def detail():
    from draw import stack
    stack('41_V4深度可分离卷积与相加', 'V4：三层Skip处理、相加与D0计算', [
        ('input', '1. 原始通道对齐后的三条Skip\nS0：[B,64,64,64]\nS1：[B,128,32,32]\nS2：[B,256,16,16]', 'input'),
        ('dw', '2. 三条支路分别做DWConv2d\n3×3；padding1；groups＝各自输入通道数\nS0步长4；S1步长2；S2步长1\n空间输出均为16×16，通道数各自保留', 'projection'),
        ('pw', '3. 各自用PWConv2d统一通道\n1×1：64→256、128→256、256→256\n三路输出均为 [B,256,16,16]\nDW、PW均无偏置；不加BN或激活', 'projection'),
        ('sum', '4. 三路直接逐元素相加\nF＝支路0输出＋支路1输出＋支路2输出\n[B,256,16,16]；不加权，不再拼接三路', 'state'),
        ('cat', '5. F替代D0原来接收的S2\n主路径：[B,512,8,8] → 转置卷积上采样\n得到 [B,256,16,16]，与F沿通道拼接\n拼接结果：[B,512,16,16]', 'state'),
        ('d0', '6. D0沿用原DoubleConv\nConv3×3：512→256 → BN → ReLU\nConv3×3：256→256 → BN → ReLU\n输出 [B,256,16,16]，继续进入D1', 'output'),
    ])


if __name__ == '__main__':
    draw()

"""E1 network and actual trained DStage; reuse the module figure renderer."""
from draw_module_figures import Diagram


def draw():
    d = Diagram('E1 · 4层Decoder + D-WPO', 2220)

    def n(key, x, y, text, color='projection'):
        return d.node(key, x, y, text, color, role=None)

    n('input', 115, 95, '输入 RGB\n3 ch · 256×256', 'input')
    n('stem', 115, 215, 'Stem\n3→48；256→64')
    d.edge('input', 'stem')
    for i, (raw, channels, size, y) in enumerate([
        (48, 64, 64, 345), (64, 128, 32, 465),
        (168, 256, 16, 585), (224, 512, 8, 705),
    ]):
        n(f'e{i}', 115, y, f'E{i} · {raw} ch\n{size}×{size}', 'input')
        n(f'a{i}', 380, y, f'1×1 对齐\n{raw}→{channels}')
        label = ('D3 · D-WPO残差块' if i == 3 else f'D{i} · DoubleConv')
        n(f'd{i}', 660, y, f'{label}\n{channels} ch · {size}×{size}',
          'state' if i == 3 else 'output')
        d.edge(f'e{i}', f'a{i}', source='right', target='left')
        d.edge(f'a{i}', f'd{i}', source='right', target='left')
        if i:
            d.edge(f'e{i-1}', f'e{i}')
            d.edge(f'd{i}', f'd{i-1}', source='top', target='bottom')
    d.edge('stem', 'e0')
    d.label(115, 280, 'Encoder')
    d.label(380, 280, '通道对齐')
    d.label(660, 280, 'Decoder')
    n('head', 660, 165, '输出头 64→1\n插值×4 → Sigmoid\n256×256 病灶概率', 'output')
    d.edge('d0', 'head', source='top', target='bottom')
    for i, y in enumerate([345, 465, 585]):
        d.label(495, y-25, f'S{i} · 拼接')
    for y in [405, 525, 645]:
        d.label(710, y, '上采样×2')

    # Dashed connector expands D3; it does not carry another feature.
    a = d.nodes['d3']
    d.ax.plot([a['x'], a['x'], 400], [a['bottom'], 800, 800],
              ls='--', lw=1, color=d.style.cfg['colors']['ink'])
    d.label(400, 835, 'D3模块展开 · 整个DoubleConv替换为下方计算')

    n('x', 400, 930, '① 对齐后的最深特征 X\n[B,512,8,8]', 'input')
    n('project', 400, 1060, '② DW3 → Linear：512→1536\n分成  $U_0$、 $V_0$、Z；各512通道')
    n('fft', 245, 1200, '③  $U_0$、 $V_0$ → rFFT2\n复数频谱：[B,512,8,5]')
    n('z', 625, 1200, 'Z · 内容门控分支\n[B,512,8,8]', 'input')
    n('wave', 245, 1370, '④ 方向化波传播\n$\\lambda=a_x k_x^2+a_y k_y^2$\n阻尼响应 + 波响应混合\n传播时间固定为1', 'state')
    n('inverse', 245, 1540, '⑤ irFFT2 → LayerNorm\n恢复 [B,512,8,8]')
    n('silu', 625, 1540, 'SiLU(Z)\n[B,512,8,8]')
    n('gate', 400, 1680, '⑥ 逐元素内容门控\n归一化结果 × SiLU(Z)', 'state')
    n('linear', 400, 1810, '⑦ 输出Linear：512→512\n逐位置通道投影')
    n('out', 400, 1960, '⑧ 逐通道缩放 + 输入残差\n$Y=X+\\gamma\\odot\\mathrm{D\\! -\\! WPO}(X)$\n[B,512,8,8]', 'output')
    for a, b in [('x', 'project'), ('project', 'fft'), ('project', 'z'),
                 ('fft', 'wave'), ('wave', 'inverse'), ('inverse', 'gate'),
                 ('z', 'silu'), ('silu', 'gate'), ('gate', 'linear'),
                 ('linear', 'out')]:
        d.edge(a, b)
    a, b = d.nodes['x'], d.nodes['out']
    d.edge('x', 'out', source='left', target='left', route=[
        (a['left'], a['y']), (30, a['y']), (30, b['y']), (b['left'], b['y']),
    ])
    d.label(85, 1810, '原始 X')
    d.label(400, 2100, '只替换D3；D2 / D1 / D0与三条Skip保持原接法')
    d.label(400, 2150, 'γ为逐通道参数，初值0.001；D-WPO全程FP32')
    d.save('28_E1_DWPO_整网与模块一体图')


if __name__ == '__main__':
    draw()

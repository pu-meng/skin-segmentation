"""Code-faithful U-shaped baseline; shared FigureStyle handles all visual style.
Layout references: 01a_完整网络与插入位置.png and 03a_完整网络与新增位置.png.
"""
from pathlib import Path
import sys
sys.path.insert(0, '/home/PuMengYu/.codex/skills/research-figure-style/scripts')
from figure_style import FigureStyle
ROOT = Path(__file__).resolve().parents[3]


def draw(current=False, *, output_stem=None, title_override=None,
         decoder_candidate=False, bottleneck_label=None):
    from matplotlib.patches import Circle
    s = FigureStyle()
    fig, ax = s.canvas(1570 if current or bottleneck_label else 1410)
    c = s.cfg['colors']
    nodes = {}
    left, middle, concat, right = 160, 407, 532, 665

    def box(key, x, y, text, color):
        node = s.box(ax, x, y, text, c[color])
        nodes[key] = node
        return node

    def vertical(a, b):
        a, b = nodes[a], nodes[b]
        start, end = ('bottom', 'top') if a['y'] < b['y'] else ('top', 'bottom')
        s.arrow(ax, [(a['x'], a[start]), (b['x'], b[end])])

    def horizontal(a, b):
        a, b = nodes[a], nodes[b]
        s.arrow(ax, [(a['right'], a['y']), (b['left'], b['y'])])

    title = 'TinyViM-S＋V2：最浅Skip-LoRA改造位置' if current else 'TinyViM-S＋UNet：基线'
    if title_override:
        title = title_override
    s.text(ax, 400, 28, title, role='heading')
    s.text(ax, left, 75, 'TinyViM-S 编码器', role='annotation')
    s.text(ax, right, 75, '统一 UNet 解码器', role='annotation')
    box('input', left, 157, 'RGB 输入\n3 ch · 256×256\nImageNet 均值/方差归一化', 'input')
    box('stem', left, 345, 'Stem\n3×3卷积，步长2，3→24\n3×3卷积，步长2，24→48\n每次卷积后 BN＋GELU', 'projection')
    vertical('input', 'stem')

    ys = [610, 850, 1090, 1330]
    widths = [48, 64, 168, 224]
    aligned = [64, 128, 256, 512]
    sequences = ['Local × 2 → TViM × 1', 'Local × 2 → TViM × 1',
                 'Local × 4 → TViM × 1\nLocal × 3 → TViM × 1', 'Local × 5 → TViM × 1']
    for i, (y, ch, target, seq) in enumerate(zip(ys, widths, aligned, sequences)):
        size = 64 // 2**i
        title = f'Stage {i}' if i < 3 else 'Stage 3 → BN'
        box(f'stage{i}', left, y, f'{title}\n{seq}\n{ch} ch · {size}×{size}', 'input')
        box(f'align{i}', middle, y, f'1×1投影\n{ch}→{target}', 'projection')
        horizontal(f'stage{i}', f'align{i}')
        if i < 3:
            box(f'down{i}', left, (ys[i]+ys[i+1])/2,
                f'下采样 ×2\n3×3卷积，步长2＋BN\n{ch}→{widths[i+1]}', 'projection')
            vertical(f'stage{i}', f'down{i}')
    vertical('stem', 'stage0')
    for i in range(3):
        vertical(f'down{i}', f'stage{i+1}')

    if current:
        box('skip_lora', middle, 475,
            '新增：Skip-LoRA r8\n64→8→64＋残差\n64 ch · 64×64', 'state')
        a, b = nodes['align0'], nodes['skip_lora']
        s.arrow(ax, [(a['x'], a['top']), (b['x'], b['bottom'])])

    # Decoder order: upsample -> concatenate aligned skip -> DoubleConv.
    for i, y in enumerate(ys[:3]):
        ch, size = aligned[i], 64 // 2**i
        if decoder_candidate and i == 1:
            box(f'decoder{i}', right, y,
                'D1 · A/C 候选\n首卷积：256→128\nA/C：128→128\n32×32', 'state')
        else:
            box(f'decoder{i}', right, y, f'D{2-i} · DoubleConv\n{2*ch}→{ch}；{size}×{size}', 'output')
        box(f'up{i}', right, y+120, f'上采样 ×2\n{2*ch}→{ch}', 'state')
        ax.add_patch(Circle((concat, y), 14, facecolor='white',
                            edgecolor=c['ink'], linewidth=s.cfg['box']['stroke_width'], zorder=3))
        s.text(ax, concat, y, 'C', role='annotation')
        s.text(ax, concat, y-33, '拼接', role='annotation')
        a, b, u = nodes[f'align{i}'], nodes[f'decoder{i}'], nodes[f'up{i}']
        if current and i == 0:
            lora = nodes['skip_lora']
            s.arrow(ax, [(lora['right'], lora['y']), (concat, lora['y']),
                         (concat, y-14)])
        else:
            s.arrow(ax, [(a['right'], y), (concat-14, y)])
        s.arrow(ax, [(concat+14, y), (b['left'], y)])
        elbow = (b['bottom'] + u['top']) / 2
        s.arrow(ax, [(right, u['top']), (right, elbow), (concat, elbow), (concat, y+14)])
    vertical('decoder2', 'up1')
    vertical('decoder1', 'up0')
    a, b = nodes['align3'], nodes['up2']
    if current or bottleneck_label:
        text = ('已有：V2瓶颈\n512 ch · 8×8\n保留原V2计算' if current else
                f'候选：{bottleneck_label}\n512 ch · 8×8\n输入、输出尺寸一致')
        box('v2', middle, 1460, text, 'state')
        v2 = nodes['v2']
        s.arrow(ax, [(a['x'], a['bottom']), (v2['x'], v2['top'])])
        s.arrow(ax, [(v2['right'], v2['y']), (right, v2['y']), (right, b['bottom'])])
    else:
        s.arrow(ax, [(a['right'], a['y']), (right, a['y']), (right, b['bottom'])])

    # Last two levels recover resolution without skips.
    box('up3', right, 510, '上采样 ×2\n64→32', 'state')
    box('decoder3', right, 415, 'D3 · DoubleConv\n32→32；128×128', 'output')
    box('up4', right, 320, '上采样 ×2\n32→16', 'state')
    box('decoder4', right, 230, 'D4 · DoubleConv\n16→16；256×256', 'output')
    box('head', right, 133, '1×1卷积＋Sigmoid\n16→1；256×256', 'output')
    for a, b in [('decoder0','up3'), ('up3','decoder3'), ('decoder3','up4'),
                 ('up4','decoder4'), ('decoder4','head')]:
        vertical(a, b)
    for i in range(3):
        if current and i == 0:
            s.text(ax, middle, ys[i]+64, 'S0 → 新模块 → 最浅拼接', role='annotation')
        else:
            s.text(ax, middle, ys[i]-62, f'skip {i} · {64//2**i}×{64//2**i}', role='annotation')
    s.compact_vertical(fig)
    stem = '21_当前TinyViM_V2_SkipLoRA架构' if current else '20_TinyViM_UNet基线架构'
    audit = s.save(fig, Path(output_stem) if output_stem else ROOT/'研究记录/figures'/stem)
    if not audit['passed']:
        raise ValueError(audit['issues'])
    print(f'Baseline figure: {len(audit["boxes"])} fitted boxes; audit passed.')


if __name__ == '__main__':
    draw()

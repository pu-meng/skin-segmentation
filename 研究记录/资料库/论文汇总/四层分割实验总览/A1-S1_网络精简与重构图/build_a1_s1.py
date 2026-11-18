from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

STYLE_SCRIPT = Path('/home/PuMengYu/.codex/skills/research-figure-style/scripts')
sys.path.insert(0, str(STYLE_SCRIPT))
from figure_style import FigureStyle

OUT = Path(__file__).parent
style = FigureStyle()
cfg = style.cfg
W = cfg['output']['width_px']
H = 3260
fig, ax = style.canvas(H)
colors = cfg['colors']


def node(x, y, text, color='input', role=None):
    return style.box(ax, x, y, text, colors[color], role=role)


def link(a, b, via=None):
    start = (a['x'], a['bottom'])
    end = (b['x'], b['top'])
    if via is None:
        style.arrow(ax, [start, end])
    else:
        style.arrow(ax, [start, (start[0], via), (end[0], via), end])


def heading(y, text):
    style.text(ax, W / 2, y, text, role='heading')


def panel(y0, y1, title):
    box = FancyBboxPatch((18, y0), W - 36, y1 - y0,
                         boxstyle='round,pad=0,rounding_size=9',
                         facecolor='#ffffff', edgecolor='#b9c7d3', linewidth=1.0,
                         zorder=0)
    ax.add_patch(box)
    heading(y0 + 25, title)


# U-shaped overview follows Encoder.forward and Decoder.forward exactly.
panel(38, 1230, 'A1-S1｜TinyViM-S 编码器—四层解码器（按代码数据流）')
style.text(ax, W / 2, 91, 'L = LocalBlock；T = TViMBlock（顺序按 Stage() 实现）', role='annotation')
inp = node(180, 127, 'RGB 输入\n[B,3,256,256]', 'input')
stem = node(235, 225, 'Stem: (ConvBN 3×3/s2 + GELU) ×2\n[B,48,64,64]', 'projection')
link(inp, stem)
s0 = node(180, 335, 'S0：L×2 → T\nraw [B,48,64,64]\nAlign 1×1：48→64 = f0', 'input')
link(stem, s0)
em1 = node(180, 445, 'Embedding：ConvBN 3×3/s2\n48→64；[B,64,32,32]', 'projection')
link(s0, em1)
s1 = node(180, 555, 'S1：L×2 → T\nraw [B,64,32,32]\nAlign 1×1：64→128 = f1', 'input')
link(em1, s1)
em2 = node(180, 665, 'Embedding：ConvBN 3×3/s2\n64→168；[B,168,16,16]', 'projection')
link(s1, em2)
s2 = node(180, 775, 'S2：L×4 → T → L×3 → T\nraw [B,168,16,16]\nAlign 1×1：168→256 = f2', 'input')
link(em2, s2)
em3 = node(180, 885, 'Embedding：ConvBN 3×3/s2\n168→224；[B,224,8,8]', 'projection')
link(s2, em3)
s3 = node(180, 995, 'S3：L×5 → T → BN\nraw [B,224,8,8]\nAlign 1×1：224→512 = f3', 'input')
link(em3, s3)
bn = node(430, 1080, 'BottleneckS1\n[B,512,8,8]', 'state')
style.arrow(ax, [(s3['right'], s3['y']), (bn['left'], bn['y'])])
d3 = node(430, 1170, 'D3S1：stages[3]\n[B,512,8,8]', 'state')
link(bn, d3)

# Decoder ascends from the bridge; skips enter the matching-resolution concat.
up2 = node(650, 1090, 'Ups[2]：512→256\n8²→16²', 'projection')
style.arrow(ax, [(d3['x'] + d3['w'] / 2, d3['top']), (up2['x'], up2['bottom'])])
c2 = node(620, 985, 'Concat(Up2, f2)\n[B,512,16,16]', 'state')
link_up = lambda a, b: style.arrow(ax, [(a['x'], a['top']), (b['x'], b['bottom'])])
link_up(up2, c2)
d2 = node(620, 885, 'D2S1：stages[2]\n512→256；[B,256,16,16]', 'output')
link_up(c2, d2)
up1 = node(620, 780, 'Ups[1]：256→128\n16²→32²', 'projection')
link_up(d2, up1)
c1 = node(620, 680, 'Concat(Up1, f1)\n[B,256,32,32]', 'state')
link_up(up1, c1)
d1 = node(620, 580, 'D1：DoubleConv\n256→128；[B,128,32,32]', 'input')
link_up(c1, d1)
up0 = node(620, 480, 'Ups[0]：128→64\n32²→64²', 'projection')
link_up(d1, up0)
c0 = node(620, 380, 'E3：双源 Skip (f0, D)\nConcat → [B,128,64,64]', 'state')
link_up(up0, c0)
d0 = node(620, 280, 'D0：DoubleConv 128→64\n[B,64,64,64]', 'input')
link_up(c0, d0)
head = node(620, 180, 'Head：Conv 1×1 64→1\n上采样256² + Sigmoid', 'output')
link_up(d0, head)

for source, target, route_y in ((s0, c0, 300), (s1, c1, 520), (s2, c2, 740)):
    style.arrow(ax, [(source['right'], source['y']), (400, source['y']),
                     (400, target['y']), (target['left'], target['y'])])

# Expanded bottleneck.
panel(1240, 1830, 'BottleneckS1｜CPE、三分支局部／全局建模、FFN 与外层残差')
b_in = node(W / 2, 1340, '输入 X\n[B,512,8,8]', 'input')
b_cpe = node(W / 2, 1420, 'CPE：X + DWConv3(X)\n[B,512,8,8]', 'projection')
link(b_in, b_cpe)
b_reduce = node(W / 2, 1500, 'PWConv 512→256；按通道切分 64／96／96', 'projection')
link(b_cpe, b_reduce)
b1 = node(145, 1600, '64 通道\nLN → Mamba\n残差回加', 'state')
b2 = node(W / 2, 1600, '96 通道\nDW3 → GN8 → GELU\nPW 96→96', 'input')
b3 = node(W - 145, 1600, '96 通道\nDW7 → GN8 → GELU\nPW 96→96', 'input')
# Branch fan-out/fan-in links.
for b in (b1, b2, b3):
    style.arrow(ax, [(b_reduce['x'], b_reduce['bottom']), (b_reduce['x'], 1555), (b['x'], 1555), (b['x'], b['top'])])
b_cat = node(W / 2, 1700, 'Concat 256 → PWConv 256→256 → FFN\nPWConv 256→128 → GN8 → GELU\nPWConv 128→256', 'projection')
for b in (b1, b2, b3):
    style.arrow(ax, [(b['x'], b['bottom']), (b['x'], 1655), (b_cat['x'], 1655), (b_cat['x'], b_cat['top'])])
b_out = node(W / 2, 1790, 'PWConv 256→512；输出 = CPE 主路 + 分支／FFN 残差\n[B,512,8,8]', 'output')
link(b_cat, b_out)

# Expanded D3.
panel(1860, 2440, 'D3S1｜RMT-style D-WPO、局部深度卷积和两级缩放残差')
d3in = node(W / 2, 1940, '输入 X\n[B,512,8,8]', 'input')
d3cpe = node(W / 2, 2015, 'CPE：X ← X + DWConv3(X)', 'projection')
link(d3in, d3cpe)
wpo = node(225, 2125, 'DirectionalWaveOperator\nDWConv3 → Linear 512→1536\n切分 u0、v0、z；rFFT2(u0)、rFFT2(v0)\n方向性传播 → irFFT2 → Norm\nSiLU(z) 门控 → Linear 512→512', 'state')
lce = node(W - 225, 2125, '局部分支\nDWConv5\n（512 通道）', 'input')
for b in (wpo, lce):
    style.arrow(ax, [(d3cpe['x'], d3cpe['bottom']), (d3cpe['x'], 2070), (b['x'], 2070), (b['x'], b['top'])])
d3mix = node(W / 2, 2250, '第一次残差：x1 = x + wave_scale × (WPO(x) + DWConv5(x))\nwave_scale：逐通道，初始化 1e-3', 'projection')
for b in (wpo, lce):
    style.arrow(ax, [(b['x'], b['bottom']), (b['x'], 2205), (d3mix['x'], 2205), (d3mix['x'], d3mix['top'])])
d3out = node(W / 2, 2390, '第二次残差：x1 + ffn_scale × FFN(x1)\nFFN：PW 512→256 → GN8 → GELU → DWConv3\n→ GELU → PW 256→512；初始化 1e-3\n输出 [B,512,8,8]', 'output')
link(d3mix, d3out)

# Expanded D2.
panel(2500, 3195, 'D2S1｜Skip 融合后共享 128 通道瓶颈的三层 DSC')
skip = node(195, 2595, 'Encoder Skip\n[B,256,16,16]', 'input')
up = node(W - 195, 2595, 'Decoder 上采样\n[B,256,16,16]', 'input')
concat = node(W / 2, 2675, 'Concat Decoder 与 Skip\n[B,512,16,16]', 'state')
for b in (skip, up):
    style.arrow(ax, [(b['x'], b['bottom']), (b['x'], 2640), (concat['x'], 2640), (concat['x'], concat['top'])])
fus = node(W / 2, 2755, 'Skip 融合：PWConv 512→256，得到 U\n[B,256,16,16]', 'projection')
link(concat, fus)
reduce = node(W / 2, 2835, '共享窄通道：PWConv 256→128', 'projection')
link(fus, reduce)
dsc1 = node(W / 2, 2915, 'DSC 1：DWConv3 → GroupNorm(8) → GELU\n[B,128,16,16]', 'input')
dsc2 = node(W / 2, 3005, 'DSC 2：DWConv3 → GroupNorm(8) → GELU\n[B,128,16,16]', 'input')
dsc3 = node(W / 2, 3105, 'DSC 3：DWConv3 → GroupNorm(8) → GELU\nPWConv 128→256；LayerScale × U 残差\nscale 初值 1e-3；输出 [B,256,16,16]', 'output')
link(reduce, dsc1)
link(dsc1, dsc2)
link(dsc2, dsc3)

ax.text(W / 2, 3225,
        'D1、D0 与 Head 保持 A1；D2／D1／D0 分别接入同尺度 Encoder Skip。',
        ha='center', va='center', fontproperties=style.regular,
        fontsize=cfg['sizes']['annotation'], color=colors['dimension'])

audit = style.save(fig, OUT / 'A1-S1_代码结构图')
print('passed:', audit['passed'])
print('svg:', OUT / 'A1-S1_代码结构图.svg')
print('issues:', audit['issues'])

from pathlib import Path
import sys

STYLE_SCRIPT = Path('/home/PuMengYu/.codex/skills/research-figure-style/scripts')
sys.path.insert(0, str(STYLE_SCRIPT))
from figure_style import FigureStyle
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'A1-S1_网络精简与重构图' / 'Skip1v06_代码结构图'
style = FigureStyle()
cfg = style.cfg
W = cfg['output']['width_px']
fig, ax = style.canvas(3200)
C = cfg['colors']
ink = C['ink']


def node(x, y, label, color='input', role='annotation'):
    return style.box(ax, x, y, label, C[color], role=role)


def line(points, arrow=True, lw=1.2, ls='-'):
    from matplotlib.patches import FancyArrowPatch
    if len(points) > 2:
        ax.plot([p[0] for p in points[:-1]], [p[1] for p in points[:-1]], color=ink, lw=lw, ls=ls, zorder=1)
    if arrow:
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>', mutation_scale=9,
                                     lw=lw, color=ink, shrinkA=0, shrinkB=0, zorder=2))
    else:
        ax.plot([points[-2][0], points[-1][0]], [points[-2][1], points[-1][1]], color=ink, lw=lw, ls=ls, zorder=1)


def center(n): return (n['x'], n['y'])
def top(n): return (n['x'], n['top'])
def bottom(n): return (n['x'], n['bottom'])
def left(n): return (n['left'], n['y'])
def right(n): return (n['right'], n['y'])


def op(x, y, symbol, color='output', radius=16):
    circ = Circle((x, y), radius, facecolor=C[color], edgecolor=ink, linewidth=1.1, zorder=3)
    ax.add_patch(circ)
    style.text(ax, x, y, symbol, role='heading')
    return {'x': x, 'y': y, 'top': y-radius, 'bottom': y+radius, 'left': x-radius, 'right': x+radius}


def section(y, title):
    style.text(ax, W/2, y, title, role='heading')
    ax.plot([25, W-25], [y+23, y+23], color='#aeb9c5', lw=.8, zorder=0)


# (a) Main flow: short operation nodes and visible branch/merge topology.
style.text(ax, W/2, 28, 'Skip1-v0.6｜按 candidate.py 的真实前向数据流绘制', role='heading')
section(72, '（a）整体数据流｜仅替换 A1-S1 的最浅层 E3')
s = node(205, 145, 'Encoder S\n[B,64,64,64]', 'input', 'dimension')
d = node(595, 145, 'Decoder D\n[B,64,64,64]', 'input', 'dimension')
h = node(205, 245, 'HaarLocalBlock\n频带局部特征 F_H', 'projection')
c = node(595, 245, 'DCTLocalBlock\nD条件频率特征 F_C', 'state')
line([bottom(s), (205, 210), top(h)])
line([bottom(s), (90, 190), (90, 245), left(h)])
line([bottom(s), (595, 190), top(c)])
line([bottom(d), (710, 190), (710, 225), right(c)])
vcat = op(400, 350, 'C', 'output')
vconv = node(400, 405, '1×1 Conv\nV：64通道', 'output')
line([bottom(h), (205, 315), (360, 315), (360, 350), left(vcat)])
line([bottom(c), (595, 315), (440, 315), (440, 350), right(vcat)])
line([bottom(vcat), top(vconv)])
sg = node(205, 350, 'S、D、S−D\n空间条件', 'input')
sdw = node(205, 425, 'DWConv 3×3\ngroups=192', 'projection')
spw = node(205, 505, 'PWConv 1×1\nA_s：[B,1,64,64]', 'projection')
line([bottom(s), (95, 145), (95, 350), left(sg)])
line([bottom(d), (740, 145), (740, 350), (sg['right']+10,350), right(sg)])
line([bottom(sg), top(sdw)]); line([bottom(sdw), top(spw)])
cg = node(595, 350, 'GAP(D) → MLP\nA_c：[B,64,1,1]', 'projection')
line([bottom(d), (760,145), (760,350), right(cg)])
gateadd = op(400, 560, '+', 'state')
sig = node(400, 620, 'Sigmoid\nG=σ(A_s+A_c)', 'state')
mul = op(400, 700, '×', 'output')
line([bottom(spw), (205,560), (left(gateadd)[0],560), left(gateadd)])
line([bottom(cg), (595,560), (right(gateadd)[0],560), right(gateadd)])
line([bottom(gateadd), top(sig)]); line([bottom(sig), top(mul)])
line([bottom(vconv), (400, 675), top(mul)])
fcross = node(400, 700, 'F_cross = V ⊙ G', 'output')
line([bottom(mul), top(fcross)])
l = node(205, 700, 'LocalRepDSC(S)\n局部保护 L_cross', 'projection')
line([bottom(s), (50,145), (50,700), left(l)])
plus = op(400, 775, '+', 'output')
line([bottom(fcross), top(plus)]); line([right(l), (310,700), (310,775), left(plus)])
delta = node(400, 835, '1×1 Conv\nΔ', 'projection')
scaled = node(595, 835, '可学习 γ\n初始化 0.01', 'state')
mul2 = op(400, 900, '×', 'state')
res = op(400, 965, '+', 'output')
line([bottom(plus), top(delta)]); line([bottom(delta), top(mul2)])
line([bottom(scaled), (595,900), right(mul2)])
line([bottom(mul2), top(res)])
line([bottom(s), (25,145), (25,965), left(res)])
spr = node(400, 1025, 'S′ = S + γΔ', 'output')
out = node(400, 1085, 'Concat(S′, D)\n[B,128,64,64] → 原 D0', 'output', 'dimension')
line([bottom(res), top(spr)]); line([bottom(spr), top(out)])
line([bottom(d), (780,145), (780,1085), right(out)])

# (b) Haar branch: subbands are shown as four parallel paths.
section(1150, '（b）HaarLocalBlock｜通道压缩、四子带并行与局部融合')
hs = node(80, 1220, 'S\n64×64×64', 'input', 'dimension')
hr = node(285, 1220, 'PW 1×1\n64→16', 'projection')
hd = node(530, 1220, 'Haar DWT\nH,W 减半', 'state')
line([right(hs), left(hr)]); line([right(hr), left(hd)])
band_x = [95, 300, 500, 705]
band_names = ['LL', 'LH', 'HL', 'HH']
band_nodes=[]
for x, name in zip(band_x, band_names):
    b = node(x, 1320, f'{name}\n16×32×32', 'input', 'dimension')
    conv = node(x, 1430, 'DW3×3\nGN\nPW1×1\nGELU', 'projection')
    band_nodes.append(b)
    line([bottom(hd), (530,1275), (x,1275), top(b)])
    line([bottom(b), top(conv)])
idwt = node(400, 1560, '逆 Haar DWT\n四子带重建', 'state')
for x in band_x:
    line([(x, 1500), (x, 1535), (400,1535), top(idwt)])
hexpand = node(400, 1625, 'PWConv 16→64', 'projection')
line([bottom(idwt), top(hexpand)])
hloc = node(125, 1625, 'LocalRepDSC(S)', 'projection')
line([bottom(hs), (35,1220), (35,1625), left(hloc)])
hadd = op(400, 1725, '+', 'output')
line([bottom(hexpand), (400,1695), top(hadd)])
line([right(hloc), (260,1625), (260,1725), left(hadd)])
hfuse = node(400, 1790, '1×1 Conv 融合\nF_H：[B,64,64,64]', 'output', 'dimension')
line([bottom(hadd), top(hfuse)])

# (c) DCT branch: decoder-conditioned frequency mask, then inverse transform.
section(1905, '（c）DCTLocalBlock｜Decoder 条件频率掩码')
ds = node(65, 1980, 'S\n64×64×64', 'input', 'dimension')
dct = node(220, 1980, 'DCT 2D', 'state')
zs = node(365, 1980, 'Z_S', 'input')
zcat = node(515, 1980, 'Concat Z_S, |Z_S|', 'projection')
af = node(690, 1980, 'PWConv\nA_f', 'projection')
line([right(ds), left(dct)]); line([right(dct), left(zs)]); line([right(zs), left(zcat)]); line([right(zcat), left(af)])
dd = node(65, 2090, 'Decoder D', 'input')
gap = node(220, 2090, 'GAP', 'projection')
mlp = node(365, 2090, 'MLP\n64→64→64', 'projection')
ac = node(515, 2090, 'A_c', 'state')
line([bottom(dd), top(gap)]); line([right(gap), left(mlp)]); line([right(mlp), left(ac)])
gadd = op(690, 2090, '+', 'state')
line([bottom(af), (690,2040), top(gadd)])
line([right(ac), left(gadd)])
tanh = node(690, 2165, 'tanh → M\n广播到 64×64', 'state')
line([bottom(gadd), top(tanh)])
freqmul = op(365, 2250, '⊙', 'output')
line([bottom(zs), (365,2045), top(freqmul)])
line([bottom(tanh), (690,2250), right(freqmul)])
idct = node(365, 2320, 'IDCT 2D', 'state')
line([bottom(freqmul), top(idct)])
dloc = node(120, 2320, 'LocalRepDSC', 'projection')
line([bottom(ds), (35,1980), (35,2320), left(dloc)])
dadd = op(365, 2400, '+', 'output')
line([bottom(idct), top(dadd)]); line([right(dloc), (220,2320), (220,2400), left(dadd)])
dfuse = node(365, 2465, '1×1 Conv 融合\nF_C：[B,64,64,64]', 'output', 'dimension')
line([bottom(dadd), top(dfuse)])

# (d) Shared local block, drawing MobileOne-like training branches and exact deploy fold.
section(2525, '（d）LocalRepDSC｜MobileOne式训练分支；推理前折叠为单个深度卷积')
lx = node(100, 2595, '输入 X\n[B,64,64,64]', 'input', 'dimension')
branches=[]
for x, label in zip([280, 480, 680], ['DWConv3×3\n+ BN', 'DWConv1×1\n+ BN', 'Identity\n+ BN']):
    branches.append(node(x, 2675, label, 'projection'))
for b in branches:
    line([right(lx), (170,2595), (170,b['y']), left(b)])
lsum=op(480, 2755, '+', 'output')
for b in branches: line([bottom(b), (b['x'],2725), (480,2725), top(lsum)])
deploy=node(680, 2755, '部署等价：DWConv3×3', 'state')
line([right(lsum), (570,2755), left(deploy)], arrow=True, ls='--')
lact=node(480, 2810, 'GELU', 'state')
lpw=node(480, 2875, 'PWConv 1×1 → BN → GELU', 'output')
ly=node(480, 2940, '输出 Y\n[B,64,64,64]', 'input', 'dimension')
line([bottom(lsum),top(lact)]);line([bottom(lact),top(lpw)]);line([bottom(lpw),top(ly)])
style.save(fig, OUT)
print(OUT.with_suffix('.svg'))

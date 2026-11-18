"""Draw the implemented E3.1 forward path using the shared FigureStyle."""
from pathlib import Path
import sys
sys.path.insert(0, '/home/PuMengYu/.codex/skills/research-figure-style/scripts')
from figure_style import FigureStyle

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '研究记录/模块设计/figures/三模块/E3_1_最浅Skip跨层选择融合'

def main():
    s = FigureStyle()
    fig, ax = s.canvas(1800)
    c = s.cfg['colors']
    s.text(ax, 400, 28, 'E3.1：Decoder 引导的最浅 Skip 修正', 'heading')
    y = 60
    nodes = {}

    def row(specs):
        nonlocal y
        heights = [s.measure(ax, text)[1] for _, _, text, _ in specs]
        height = max(heights)
        for key, x, text, color in specs:
            nodes[key] = s.box(ax, x, y + height/2, text, c[color])
        y += height + 38

    row([('where', 400, '① 接入位置\nD1 → 上采样 → D；Encoder 最浅层 → S\n两者均为 [B,64,64,64]', 'input')])
    row([('S', 205, '② Skip 输入 S\n[B,64,64,64]', 'input'),
         ('D', 570, 'Decoder 输入 D\n[B,64,64,64]', 'input')])
    row([('A', 205, '③ 全局路：FullScan\n1×1 Conv＋BN\nDW3＋SiLU\n四向扫描 → 还原并合并\nLN → 1×1 Conv＋BN\nA：[B,64,64,64]', 'projection'),
         ('B', 570, '④ 局部路：并行卷积\n同一份 S 分别进入\nRepDW7 ／ 官方 FDConv3\n拼接：[B,128,64,64]\n1×1 Conv：128 → 64\nB：[B,64,64,64]', 'state')])
    row([('gates', 400, '⑤ 两路互相门控\n$A_g=\mathrm{Gate}(A,B)$\n$B_g=\mathrm{Gate}(B,A)$\n两门均读取更新前的 A、B', 'projection')])
    row([('select', 400, '⑥ Decoder 逐位置选择两路\n$(w_0,w_1)=\mathrm{softmax}_{2}(\mathrm{Conv}_{1}(D))$\n$M=w_0\odot A_g+w_1\odot B_g$\n权重 [B,2,64,64]；向64通道广播', 'state')])
    row([('res', 400, '⑦ 加入 Decoder 投影，残差修正 Skip\n$S_{\mathrm{new}}=S+\eta\odot(M+\mathrm{Conv}_{1}(D))$\nη：[1,64,1,1]；初值 0.001', 'output')])
    row([('concat', 400, '⑧ 与原 Decoder 特征拼接\n$\mathrm{Concat}(D,S_{\mathrm{new}})$\n[B,128,64,64] → 原 D0 → 分割头', 'output')])

    def edge(a,b,offset=0):
        u,v=nodes[a],nodes[b]
        s.arrow(ax,[(u['x']+offset,u['bottom']), (u['x']+offset,(u['bottom']+v['top'])/2),
                    (v['x'],(u['bottom']+v['top'])/2), (v['x'],v['top'])])

    edge('S','A')
    u,v=nodes['S'],nodes['B']
    fork=u['bottom']+18
    s.arrow(ax,[(u['x'],u['bottom']),(u['x'],fork),(v['x'],fork),(v['x'],v['top'])])
    # Independent A and B outputs enter the joint gate stage.
    for a,x in [('A',300),('B',500)]:
        u,v=nodes[a],nodes['gates']
        s.arrow(ax,[(u['x'],u['bottom']),(u['x'],v['top']-18),(x,v['top']-18),(x,v['top'])])
    for a,b in [('gates','select'),('select','res'),('res','concat')]:edge(a,b)
    # D stays unchanged; a right-side bus feeds selection, projection and concat.
    u=nodes['D'];bus=765
    for key in ['select','res','concat']:
        v=nodes[key]
        s.arrow(ax,[(u['right'],u['y']),(bus,u['y']),(bus,v['y']),(v['right'],v['y'])])
    # Original S is the outer residual of the refiner.
    u,v=nodes['S'],nodes['res']
    s.arrow(ax,[(u['left'],u['y']),(35,u['y']),(35,v['y']),(v['left'],v['y'])])
    s.text(ax, 400, y-4, '实线按实际 forward；只输出修正后的 Skip，D 保留原值。', 'annotation')
    height=y+22
    fig.set_size_inches(800/s.cfg['output']['dpi'],height/s.cfg['output']['dpi'])
    ax.set_ylim(height,0)
    s.save(fig,OUT)
    print(OUT.with_suffix('.svg'))

if __name__ == '__main__': main()

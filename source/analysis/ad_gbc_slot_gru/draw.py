"""Slot Attention and its GRU, traced from pinned official source snapshots.

Shared FigureStyle owns all typography, colors, fitted boxes and previews.
"""
from pathlib import Path
import sys
sys.path.insert(0, '/home/PuMengYu/.codex/skills/research-figure-style/scripts')
from figure_style import FigureStyle

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'研究记录/figures'


class Diagram:
    def __init__(self, title):
        self.s = FigureStyle()
        self.fig, self.ax = self.s.canvas()
        self.nodes = {}
        self.s.text(self.ax, 400, 27, title, role='heading')

    def box(self, key, x, top, text, color):
        w, h = self.s.measure(self.ax, text)
        n = self.s.box(self.ax, x, top+h/2, text, self.s.cfg['colors'][color])
        self.nodes[key] = n
        return n

    def arrow(self, points):
        self.s.arrow(self.ax, points)

    def link(self, a, b):
        a, b = self.nodes[a], self.nodes[b]
        mid = (a['bottom'] + b['top'])/2
        self.arrow([(a['x'],a['bottom']), (a['x'],mid), (b['x'],mid), (b['x'],b['top'])])

    def save(self, name):
        h = max(n['bottom'] for n in self.nodes.values()) + 35
        width, dpi = self.s.cfg['output']['width_px'], self.s.cfg['output']['dpi']
        self.fig.set_size_inches(width/dpi, h/dpi)
        self.ax.set_ylim(h, 0)
        audit = self.s.save(self.fig, OUT/name)
        if not audit['passed']:
            raise ValueError(audit['issues'])
        print(name, 'passed', flush=True)


def slots():
    d = Diagram('Slot Attention：竞争分配 → 汇聚 → 迭代更新槽位')
    a = d.box('kv', 205, 72,
        '1. 输入特征归一化与投影\nX：[B,N,C] → LN → 线性k、v\nk、v：[B,N,D]；迭代间复用', 'input')
    b = d.box('init', 602, 72,
        '2. 高斯初始化槽位\n'+r'$S^{(0)}=\mu+\exp(\ell)\odot\eta$'
        +'\n[B,K,D]；μ、ℓ可学习', 'input')
    assert a['right'] + 12 < b['left'] and b['right'] < 795
    top = max(a['bottom'],b['bottom']) + 42
    specs = [
        ('q', '3. 保存旧槽位，再归一化生成q\n'
         +r'$S_{\rm prev}=S^{(t-1)}$'+'\n'
         +r'$q=\operatorname{LN}(S_{\rm prev})W_q$'
         +'\nSprev、q：[B,K,D]', 'projection'),
        ('score', '4. 每个输入与每个槽位计算匹配分数\n'
         +r'$M[b,i,k]=D^{-1/2}\sum_d k[b,i,d]q[b,k,d]$'
         +'\nM：[B,N,K]；沿特征轴d求和', 'projection'),
        ('softmax', '5. 沿K个槽位做Softmax：槽位竞争\n'
         +r'$A[b,i,k]=\operatorname{softmax}_{k}(M[b,i,k])$'
         +'\n每个输入i分给所有槽位的权重之和为1', 'state'),
        ('norm', '6. 加ε，再沿N个输入归一化\n'
         +r'$\widehat{A}=(A+\epsilon)/\operatorname{sum}_{N}(A+\epsilon)$'
         +'\n每个槽位k接收所有输入的权重之和为1', 'state'),
        ('updates', '7. 各槽位加权汇总v\n'
         +r'$U[b,k,d]=\sum_i\widehat{A}[b,i,k]v[b,i,d]$'
         +'\nU：[B,K,D]；沿N个输入求和', 'state'),
        ('gru', '8. GRU接收新汇总与旧槽位\n'
         +r'$G=\operatorname{GRUCell}(U,S_{\rm prev})$'
         +'\n每张图、每个槽位独立更新；共享GRU参数', 'projection'),
        ('mlp', '9. 归一化 → 两层MLP → 残差\n'
         +r'$S^{(t)}=G+\operatorname{MLP}(\operatorname{LN}(G))$'
         +'\nMLP：D→隐藏维度→ReLU→D\n下一轮用更新后的槽位重新生成q', 'projection'),
        ('out', '完成T轮，输出槽位特征\n'+r'$S^{(T)}$'
         +'：[B,K,D]\n这里输出K个特征向量', 'output'),
    ]
    for key, text, color in specs:
        n = d.box(key, 400, top, text, color)
        assert n['left'] > 65 and n['right'] < 705
        top = n['bottom'] + d.s.cfg['box']['node_gap']
    d.link('init','q')
    keys = [r[0] for r in specs]
    for x,y in zip(keys, keys[1:]):
        d.link(x,y)
    # Static K and V branch to the two operations consuming them.
    for target, symbol in [('score','k'),('updates','v')]:
        n = d.nodes[target]
        d.arrow([(a['left'],a['y']), (35,a['y']), (35,n['y']), (n['left'],n['y'])])
        d.s.text(d.ax, n['left']-20,n['y']-18,'$'+symbol+'$',role='annotation')
    q, gru, mlp = (d.nodes[k] for k in ('q','gru','mlp'))
    d.arrow([(q['right'],q['y']), (735,q['y']), (735,gru['y']), (gru['right'],gru['y'])])
    d.s.text(d.ax, 724, gru['y']-22, r'$S_{\rm prev}$', role='annotation')
    d.arrow([(mlp['right'],mlp['y']), (775,mlp['y']), (775,q['y']-18), (q['right'],q['y']-18)])
    d.s.text(d.ax, 753, mlp['y']-28, '再迭代', role='annotation')
    d.save('42_SlotAttention竞争汇聚与GRU更新')


def gru():
    d = Diagram('GRUCell：每张图、每个槽位的一次状态更新')
    a = d.box('input', 400, 72,
        '1. 新汇总U与上一轮槽位Sprev\n'+r'$u=U,\quad h_{\rm old}=S_{\rm prev}$'
        +'\n两者均为 [B,K,D]；这里t表示槽位更新轮次', 'input')
    p = d.box('proj', 400, a['bottom']+28,
        '2. 两组可学习线性投影，分别拆成三份\n'
        +r'$[a_z,a_r,a_n]=uW_x+b_x$'+'\n'
        +r'$[v_z,v_r,v_n]=h_{\rm old}W_h+b_h$'
        +'\nWx、Wh：[D,3D]；每份输出 [B,K,D]', 'projection')
    top = p['bottom']+35
    r = d.box('r', 235, top, '3. 重置门r\n'+r'$r=\operatorname{sigmoid}(a_r+v_r)$'
        +'\n每个槽位、每个通道一个门值', 'state')
    z = d.box('z', 565, top, '4. 更新门z\n'+r'$z=\operatorname{sigmoid}(a_z+v_z)$'
        +'\n控制保留旧状态的比例', 'state')
    assert r['right']+10 < z['left']
    d.link('proj','r'); d.link('proj','z')
    n = d.box('candidate', 400, max(r['bottom'],z['bottom'])+40,
        '5. 候选状态：先投影旧状态，再乘重置门\n'
        +r'$n=\tanh(a_n+r\odot v_n)$'
        +'\nreset_after=True；n：[B,K,D]', 'state')
    d.link('r','candidate')
    mix = d.box('mix', 400, n['bottom']+35,
        '6. 更新门混合旧状态与候选状态\n'
        +r'$h_{\rm new}=z\odot h_{\rm old}+(1-z)\odot n$'
        +'\nz接近1：保留旧状态；接近0：采用候选状态', 'output')
    d.link('candidate','mix')
    d.arrow([(z['right'],z['y']), (720,z['y']), (720,mix['y']), (mix['right'],mix['y'])])
    d.s.text(d.ax, 701, mix['y']-22, '$z$', role='annotation')
    d.arrow([(a['right'],a['y']), (775,a['y']), (775,mix['y']+20), (mix['right'],mix['y']+20)])
    d.s.text(d.ax, 742, mix['y']+43, r'$h_{\rm old}$', role='annotation')
    d.arrow([(p['left'],p['y']), (40,p['y']), (40,n['y']), (n['left'],n['y'])])
    d.s.text(d.ax, n['left']-30,n['y']-22, r'$a_n,v_n$',role='annotation')
    d.box('out', 400, mix['bottom']+35,
        '回到Slot Attention的第8块\n'+r'$G=h_{\rm new}$'
        +'：[B,K,D]\n随后执行第9块MLP残差', 'output')
    d.link('mix','out')
    d.save('43_GRU重置门更新门与候选状态')


if __name__ == '__main__':
    slots()
    gru()

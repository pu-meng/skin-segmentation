"""MoE reading guide. Structure sources are recorded in the adjacent Markdown.
Run with /home/PuMengYu/anaconda3/envs/medseg/bin/python.
SVG is the deliverable; PNG previews stay in /tmp/moe_model_guide_preview.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures'
PREVIEW = Path('/tmp/moe_model_guide_preview')
FONT = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
plt.rcParams.update({'svg.fonttype': 'path', 'svg.hashsalt': 'moe-model-guide-20261015'})
INK = '#28384A'
C = {'base': '#f6e8a6', 'attn': '#cfe3f6', 'moe': '#e6d5f7',
     'expert': '#fce5ce', 'out': '#cdefd8', 'neutral': '#e7e9ed'}
EDGES = {'base': '#a66a00', 'attn': '#2f6fa3', 'moe': '#7e57a6',
         'expert': '#a86628', 'out': '#2e7d5b', 'neutral': '#667085'}

class Drawing:
    def __init__(self, w, h, title, subtitle=''):
        self.w, self.h = w, h
        self.fig, self.ax = plt.subplots(figsize=(w*.68, h*.68))
        self.fig.subplots_adjust(0, 0, 1, 1)
        self.ax.set(xlim=(0, w), ylim=(0, h))
        self.ax.axis('off')
        self.boxes = []
        self.text(w/2, h-.48, title, 24)
        if subtitle:
            self.text(w/2, h-1.05, subtitle, 18)

    def text(self, x, y, s, size=15, color=INK):
        return self.ax.text(x, y, s, fontproperties=FONT, fontsize=max(size,18),
                            ha='center', va='center', color=color,
                            linespacing=1.3, zorder=6)

    def box(self, x, y, title, sub='', kind='base', w=3.3, h=.95):
        p = FancyBboxPatch((x-w/2, y-h/2), w, h,
                          boxstyle='round,pad=.02,rounding_size=.09',
                          facecolor=C[kind], edgecolor=EDGES[kind], lw=1.35, zorder=3)
        self.ax.add_patch(p)
        ts = [self.text(x, y+.22 if sub else y, title, 24)]
        if sub:
            ts.append(self.text(x, y-.26, sub, 20))
        self.boxes.append((p, ts))

    def arrow(self, a, b, control=False):
        self.ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>',
                         mutation_scale=16, shrinkA=0, shrinkB=0,
                         color=EDGES['moe'] if control else INK,
                         lw=1.8, linestyle='--' if control else '-', zorder=2))

    def path(self, pts, control=False):
        for a, b in zip(pts[:-2], pts[1:-1]):
            self.ax.plot([a[0], b[0]], [a[1], b[1]],
                         color=EDGES['moe'] if control else INK,
                         lw=1.35, ls='--' if control else '-', zorder=1)
        self.arrow(pts[-2], pts[-1], control)

    def plus(self, x, y):
        self.ax.add_patch(Circle((x, y), .20, facecolor='white', edgecolor=INK, lw=1.35, zorder=3))
        self.text(x, y, '+', 18)

    def save(self, name):
        OUT.mkdir(exist_ok=True)
        PREVIEW.mkdir(exist_ok=True)
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        for p, ts in self.boxes:
            bb = p.get_window_extent(renderer)
            for t in ts:
                tb = t.get_window_extent(renderer)
                if not (bb.x0 <= tb.x0 and bb.x1 >= tb.x1 and bb.y0 <= tb.y0 and bb.y1 >= tb.y1):
                    raise RuntimeError('Text exceeds box: '+t.get_text())
        for t in self.ax.texts:
            bb = t.get_window_extent(renderer)
            if bb.x0 < 0 or bb.y0 < 0 or bb.x1 > self.fig.bbox.x1 or bb.y1 > self.fig.bbox.y1:
                raise RuntimeError('Text exceeds canvas: '+t.get_text())
        path = OUT / (name+'.svg')
        self.fig.savefig(path, facecolor='white', metadata={'Date': None})
        ET.parse(path)
        self.fig.savefig(PREVIEW/(name+'.png'), dpi=100, facecolor='white')
        plt.close(self.fig)
        print(path)

def placement():
    models = [
        (2.9, 'Qwen3-30B-A3B', '全部 48 层含 MoE', [('第 1 层','注意力＋MoE'), ('第 2–47 层','各层都有注意力＋MoE'), ('第 48 层','注意力＋MoE')]),
        (8.0, 'GPT-OSS-20B', '全部 24 层含 MoE', [('第 1 层','注意力＋MoE'), ('第 2–23 层','各层都有注意力＋MoE'), ('第 24 层','注意力＋MoE')]),
        (13.1, 'DeepSeek-V2-Lite', '第 2–27 层含 MoE', [('第 1 层','注意力＋普通前馈'), ('第 2–26 层','各层都有注意力＋MoE'), ('第 27 层','注意力＋MoE')]),
        (18.2, 'OLMoE-1B-7B', '全部 16 层含 MoE', [('第 1 层','注意力＋MoE'), ('第 2–15 层','各层都有注意力＋MoE'), ('第 16 层','注意力＋MoE')])]
    for panel in range(2):
        d = Drawing(11, 14, f'图1({"ab"[panel]})｜MoE 放在哪些层？',
                    '只看位置：紫色层里面有 MoE')
        for j, (_, title, sub, layers) in enumerate(models[panel*2:panel*2+2]):
            x = 2.9+j*5.1
            d.text(x, 12.15, title, 21)
            d.text(x, 11.55, sub, 20, EDGES['moe'])
            d.box(x, 10.4, '文本变成数字特征', '', 'neutral', w=4.7)
            last = 10.4
            for i, (layer, info) in enumerate(layers):
                y = 8.55-i*2.05
                d.box(x, y, layer, info, 'base' if (panel==1 and j==0 and i==0) else 'moe', w=4.7)
                d.arrow((x,last-.475), (x,y+.475))
                last = y
            d.box(x, 2.4, '预测下一个文字单位', '', 'out', w=4.7)
            d.arrow((x,last-.475), (x,2.875))
        d.text(5.5, 1.05, '中间框代表连续多层，不是合成一层', 20)
        d.text(5.5, .45, '专家数量另外讲；这里仅标出位置', 20)
        d.save('MoE_01'+ 'ab'[panel] +'_模型位置')

def transformer():
    d = Drawing(10, 17, '图2｜MoE 在一层里的位置',
                'Qwen3 示例：注意力之后做 MoE')
    x = 5
    nodes = [(15,'上一层输出','[B,T,2048]','neutral'),
             (13.6,'RMSNorm','归一化，不改形状','neutral'),
             (12.2,'因果自注意力','汇总当前及前文信息','attn'),
             (9.3,'RMSNorm','归一化，不改形状','neutral'),
             (7.9,'MoE 前馈部分','128 选 8','moe'),
             (3.6,'本层输出','[B,T,2048] → 下一层','out')]
    for y,t,s,c in nodes: d.box(x,y,t,s,c,w=5.2)
    d.arrow((x,14.525),(x,14.075))
    d.arrow((x,13.125),(x,12.675))
    d.plus(x,10.8)
    d.arrow((x,11.725),(x,11.0))
    d.path([(x,14.3),(1.8,14.3),(1.8,10.8),(4.8,10.8)])
    d.text(1.5,12.2,'残差',14)
    d.arrow((x,10.6),(x,9.775))
    d.arrow((x,8.825),(x,8.375))
    d.plus(x,5.8)
    d.arrow((x,7.425),(x,6.0))
    d.path([(x,10.1),(8.2,10.1),(8.2,5.8),(5.2,5.8)])
    d.text(8.5,7.9,'残差',14)
    d.arrow((x,5.6),(x,4.075))
    d.text(x,1.9,'专家输入：一个词元的特征向量',20)
    d.text(x,1.2,'专家输出：变换后的特征向量',20)
    d.text(x,.5,'B：批量；T：词元数；2048：特征维度',18)
    d.save('MoE_02_单层位置')

def sparse_shared():
    for ox, shared in [(0,False),(0,True)]:
        d = Drawing(10, 14, '图3(b)｜DeepSeek：另有共享分支' if shared else '图3(a)｜Qwen3：选专家再合并',
                    '实线：特征；虚线：编号或权重')
        d.text(ox+5,12.2,'DeepSeek-V2-Lite' if shared else 'Qwen3-30B-A3B',22)
        d.box(ox+5,10.8,'归一化后的词元特征','[B,T,2048]','neutral',w=4.8)
        d.box(ox+2.8,8.5,'专家选择器','2048 → 64 分数' if shared else '2048 → 128 分数','moe',w=4.3)
        d.path([(ox+5,10.325),(ox+5,9.65),(ox+2.8,9.65),(ox+2.8,8.975)])
        d.box(ox+2.8,6.7,'选编号与权重','64 个中选 6 个' if shared else '128 个中选 8 个','moe',w=4.3 if shared else 3.8)
        d.arrow((ox+2.8,8.025),(ox+2.8,7.175),True)
        d.box(ox+7.4,6.7,'共享专家分支' if shared else '选中专家并计算',
              '2 个，每次都算' if shared else '8 个，各 2048 维','expert',w=4.3 if shared else 3.8)
        d.path([(ox+5,9.65),(ox+7.4,9.65),(ox+7.4,7.175)])
        if shared:
            d.box(ox+2.8,4.7,'6 个专家合并','得到 2048 维','expert',w=4.3)
            d.arrow((ox+2.8,6.225),(ox+2.8,5.175),True)
            d.path([(ox+5,9.65),(ox+.35,9.65),(ox+.35,4.7),(ox+.65,4.7)])
            d.plus(ox+5,2.9)
            d.path([(ox+2.8,4.225),(ox+2.8,2.9),(ox+4.8,2.9)])
            d.path([(ox+7.4,6.225),(ox+7.4,2.9),(ox+5.2,2.9)])
            d.box(ox+5,1.4,'MoE 输出','2048 维','out',w=4.8)
            d.arrow((ox+5,2.7),(ox+5,1.875))
        else:
            d.arrow((ox+4.7,6.7),(ox+5.5,6.7),True)
            d.box(ox+5,4.2,'8 个输出加权求和','2048 维','moe',w=4.8)
            d.path([(ox+7.4,6.225),(ox+7.4,5.1),(ox+5,5.1),(ox+5,4.675)])
            d.path([(ox+2.8,6.225),(ox+2.8,5.1),(ox+1.9,5.1),(ox+1.9,4.2),(ox+2.6,4.2)],True)
            d.box(ox+5,1.4,'MoE 输出','2048 维','out',w=4.8)
            d.arrow((ox+5,3.725),(ox+5,1.875))
        d.save('MoE_03b_共享分支' if shared else 'MoE_03a_路由分支')

def medical():
    d = Drawing(18, 18, '图4｜你的 MedNeXt：MoE 集中在瓶颈',
                'MLA/MHA 瓶颈版本；128³ 输入示例')
    ex, dx = 3.4, 14.5
    d.box(ex,15.9,'CT → 输入投影','1 → 32 通道','neutral',w=4.8)
    ys = [14.2,11.9,9.6,7.3]
    for i,y in enumerate(ys):
        c,sz=32*2**i,128//2**i
        d.box(ex,y,f'Encoder {i}',f'{c} 通道，{sz}³','base',w=4.2)
        if i==0: d.arrow((ex,15.425),(ex,y+.475))
        else:
            d.box(ex,y+1.15,f'Down {i-1}：减半','','neutral',w=3.6,h=.75)
            d.arrow((ex,ys[i-1]-.475),(ex,y+1.525))
            d.arrow((ex,y+.775),(ex,y+.475))
        d.box(dx,y,f'Decoder {i}',f'{c} 通道，{sz}³','base',w=4.2)
        d.plus(10.7,y)
        d.arrow((ex+2.1,y),(10.5,y))
        d.text(8.3,y+.33,f'skip {i}',13)
        d.arrow((10.9,y),(dx-2.1,y))
        if i<3:
            uy=y-1.15
            d.box(dx,uy,f'Up {i}：加倍','','neutral',w=3.6,h=.75)
            d.arrow((dx,ys[i+1]+.475),(dx,uy-.375))
            d.path([(dx,uy+.375),(dx,y-.7),(10.7,y-.7),(10.7,y-.2)])
    d.box(ex,5.8,'Down 3：到 8³','','neutral',w=4.2,h=.85)
    d.arrow((ex,6.825),(ex,6.225))
    d.box(ex,4.35,'卷积瓶颈','512 通道，8³','base',w=4.8)
    d.arrow((ex,5.375),(ex,4.825))
    d.box(ex,2.8,'上下文 Block ×2','每个：注意力＋MoE','moe',w=5.5,h=1.15)
    d.arrow((ex,3.875),(ex,3.375))
    d.box(dx,5.8,'Up 3：到 16³','','neutral',w=4.2,h=.85)
    d.path([(ex,2.225),(ex,1.4),(dx,1.4),(dx,5.375)])
    d.path([(dx,6.225),(dx,6.6),(10.7,6.6),(10.7,7.1)])
    d.box(dx,15.9,'分割输出头','背景 / 肝脏 / 肿瘤','out',w=4.8)
    d.arrow((dx,14.675),(dx,15.425))
    d.text(9.2,4.4,'MLA 版：低秩注意力\nMHA 版：标准多头注意力',14)
    d.text(9.2,3.1,'词元＝三维空间位置\n每个位置有 512 维特征',18)
    d.text(9,.55,'紫色部分含两个 MoE；其余编码、解码阶段仍用卷积',20)
    d.save('MoE_04_当前分割模型位置')

def local_moe():
    d = Drawing(19, 17, '图5｜你的 MoE：先算全部专家，再选两个输出',
                '实线：特征或原始分数；虚线：编号或权重')
    d.box(10,15,'瓶颈中归一化后的特征','[B,N,512] → [B×N,512]','neutral',w=5.6)
    d.box(3.6,12.5,'Router 线性打分','512 → 4 个原始分数','moe',w=5.1)
    d.box(11,12.5,'全部 4 个路由专家','每个 512 → 1024 → 512','expert',w=5.6)
    d.box(16.8,12.5,'共享专家','始终计算','expert',w=3.6)
    for x in [3.6,11,16.8]:
        d.path([(10,14.525),(10,13.9),(x,13.9),(x,12.975)])
    d.box(3.6,9.7,'分数＋均衡偏置','选择 Top-2 编号','moe',w=5.1)
    d.arrow((3.6,12.025),(3.6,10.175))
    d.box(11,9.7,'按编号取 2 个输出','[B×N,2,512]','neutral',w=5.6)
    d.arrow((11,12.025),(11,10.175))
    d.arrow((6.15,9.7),(8.2,9.7),True)
    d.text(7.2,10.1,'编号',13,EDGES['moe'])
    d.box(3.6,6.8,'取 2 个原始分数','softmax → 2 个权重','moe',w=5.1)
    d.arrow((3.6,9.225),(3.6,7.275),True)
    d.path([(3.6,11.3),(.5,11.3),(.5,6.8),(1.05,6.8)])
    d.text(1.65,8.25,'原始分数',13)
    d.box(11,6.8,'两个选中输出加权求和','[B×N,512]','moe',w=5.6)
    d.arrow((11,9.225),(11,7.275))
    d.arrow((6.15,6.8),(8.2,6.8),True)
    d.text(7.2,7.2,'权重',13,EDGES['moe'])
    d.plus(11,4.5)
    d.arrow((11,6.325),(11,4.7))
    d.path([(16.8,12.025),(16.8,4.5),(11.2,4.5)])
    d.box(11,2.7,'共享输出＋路由输出','还原为 [B,N,512]','out',w=5.6)
    d.arrow((11,4.3),(11,3.175))
    d.text(9.5,1.3,'N 是空间位置数；8³ 时 N = 512；各专家参数独立',20)
    d.text(9.5,.6,'实际计算：4 个路由专家＋1 个共享专家',22)
    d.save('MoE_05_当前分割MoE内部')

def purpose():
    d = Drawing(11, 12, '图6｜MoE 主要省什么？',
                '相对全算专家省计算，不省总参数')
    for x, sparse in [(2.9,False),(8.0,True)]:
        d.box(x,9.8,'稀疏 MoE' if sparse else '普通前馈',
              '存放多组专家参数' if sparse else '一套前馈参数','moe' if sparse else 'base',w=4.7)
        d.box(x,7.6,'一个词元到来',
              '从 128 个中选 8 个' if sparse else '使用同一套参数','neutral',w=4.7)
        d.arrow((x,9.325),(x,8.075))
        d.box(x,5.4,'执行选中的专家' if sparse else '执行普通前馈',
              '其余专家不为它计算' if sparse else '输出变换后的特征','expert' if sparse else 'base',w=4.7)
        d.arrow((x,7.125),(x,5.875))
    d.box(5.5,3.2,'专家参数仍需全部存储', '只算选中的专家，避免每次全算','out',w=9.5,h=1.1)
    d.text(5.5,1.65,'示意图：两列不代表一次等算力实验',20)
    d.text(5.5,.95,'8/128 是专家调用比例，不是整网算量比例',18)
    d.text(5.5,.35,'MoE 早于 DeepSeek；加上 MoE 不保证涨分',20)
    d.save('MoE_06_为什么使用')

def diagnosis():
    d = Drawing(11, 12, '图7｜你的 MoE 没涨分：先区分两件事',
                '计算方式已核对；性能原因还需要证据')
    d.box(5.5,9.8,'已确认：没有只计算 Top-2',
          '4 个路由专家＋1 个共享专家全部计算','neutral',w=9.5,h=1.1)
    d.text(5.5,8.45,'这解释计算成本，不直接解释 Dice',22)
    d.box(5.5,6.9,'① 单独换 MoE，效果改善吗？',
          '固定注意力、数据和训练，比较 MLP / MoE','base',w=9.5,h=1.1)
    d.box(5.5,4.65,'② 不同专家真的互补吗？',
          '专家参数不同，不代表能力有用或互补','expert',w=9.5,h=1.1)
    d.box(5.5,2.4,'③ 路由器能选到更好的组合吗？',
          '固定模型试换同数量专家，检查损失变化','moe',w=9.5,h=1.1)
    d.arrow((5.5,6.35),(5.5,5.2))
    d.arrow((5.5,4.1),(5.5,2.95))
    d.text(5.5,.7,'②③是待验证问题；这次没有启动新实验',20)
    d.save('MoE_07_为什么未必提升')

if __name__ == '__main__':
    placement()
    transformer()
    sparse_shared()
    medical()
    local_moe()
    purpose()
    diagnosis()
